""" Huggingface CLIP Warapper """
import os
import logging
from typing import List, Dict

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def to_batch(inputs: Dict, batch_size: int = None):
    size = len(list(inputs.values())[0])
    batch_size = size if batch_size is None or batch_size > size else batch_size
    block = list(range(0, size, batch_size)) + [size]
    batch_data = []
    for s, e in zip(block[:-1], block[1:]):
        batch_data.append({k: v[s:e] for k, v in inputs.items()})
    return batch_data


class CLIP:
    """ Huggingface CLIP Wrapper """

    def __init__(self, model: str = 'openai/clip-vit-large-patch14-336'):
        """ Huggingface CLIP Warapper

        :param model: CLIP model on huggingface
            - 'openai/clip-vit-large-patch14'
            - 'openai/clip-vit-base-patch32'
            - 'openai/clip-vit-large-patch14-336'
            - 'openai/clip-vit-base-patch16'
        """
        self.model = CLIPModel.from_pretrained(model).eval()
        self.processor = CLIPProcessor.from_pretrained(model)
        self.config = self.model.config.to_dict()
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        assert not self.parallel, "Processing on multiple GPUs is not supported"
        # if self.parallel:
        #     self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

        logging.info('** LOAD MODEL ** ')
        logging.info(f'\tDevice: {self.device} ({torch.cuda.device_count()} gpus)')
        logging.info(f"\tModel parameters: {np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        logging.info(f"\tInput resolution: {self.config['vision_config']['image_size']}")
        logging.info(f"\tContext length: {self.config['text_config']['max_position_embeddings']}")
        logging.info(f"\tVocab size: {self.config['text_config']['vocab_size']}")

    def get_similarity(self, images: List or str, texts: List or str, batch_size: int = None):
        """ get embedding

        :param images: a list of images to get embedding
        :param texts: a list of texts to get embedding
        :param batch_size: batch size
        :return: (output_image_embedding, output_text_embedding, sim)
            - output_image_embedding: a tensor of image embedding (image size x output dim)
            - output_text_embedding: a tensor of text embedding (text size x output dim)
            - sim: a tensor of similarity (image size x text size)
        """
        images = [images] if type(images) is str else images
        texts = [texts] if type(texts) is str else texts
        logging.debug(f'model inference on images: {len(images)}')
        pil_images = [Image.open(i).convert("RGB") for i in images]
        image_inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        batch_image_inputs = to_batch(image_inputs, batch_size=batch_size)
        with torch.no_grad():
            output_image_embedding = []
            for i in batch_image_inputs:
                output_image_embedding.append(
                    self.model.get_image_features(**{k: v.to(self.device) for k, v in i.items()})
                )
            output_image_embedding = torch.cat(output_image_embedding)
        logging.debug(f'model inference on texts: {len(texts)}')
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        batch_text_inputs = to_batch(text_inputs, batch_size=batch_size)
        with torch.no_grad():
            output_text_embedding = []
            for i in batch_text_inputs:
                output_text_embedding.append(
                    self.model.get_text_features(**{k: v.to(self.device) for k, v in i.items()})
                )
        output_text_embedding = torch.cat(output_text_embedding)
        logging.debug('compute similarity')
        sim = self.cos(
            output_image_embedding.unsqueeze(1).repeat((1, len(output_text_embedding), 1)),
            output_text_embedding.unsqueeze(0).repeat((len(output_image_embedding), 1, 1))
        ) * 100  # image size x text size
        return sim.cpu().numpy().tolist()

