""" Huggingface CLIP Warapper """
import os
import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def to_batch(inputs: List, batch_size: int = None):
    batch_size = len(inputs) if batch_size is None or batch_size > len(inputs) else batch_size
    block = list(range(0, len(inputs), batch_size)) + [len(inputs)]
    return [inputs[s:e] for s, e in zip(block[:-1], block[1:])]


def cosine_similarity(a, b, zero_vector_mask: float = -100):
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    if norm_b * norm_a == 0:
        return zero_vector_mask
    return sum(map(lambda x: x[0] * x[1], zip(a, b)))/(norm_a * norm_b)


class MultilingualCLIP:
    """ Huggingface CLIP Wrapper """

    def __init__(self, model: str = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'):
        """ Huggingface CLIP Warapper

        :param model: model name
        """
        self.img_model = SentenceTransformer('clip-ViT-B-32')
        self.text_model = SentenceTransformer(model)
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        assert not self.parallel, "Processing on multiple GPUs is not supported"
        for model in [self.img_model, self.text_model]:
            model.eval()
            model.to(self.device)

        logging.info('** LOAD MODEL ** ')
        logging.info(f'\tDevice: {self.device} ({torch.cuda.device_count()} gpus)')

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
        batch = to_batch([Image.open(i).convert("RGB") for i in images], batch_size=batch_size)
        with torch.no_grad():
            output_image_embedding = []
            for i in batch:
                output_image_embedding += self.img_model.encode(i).tolist()

        logging.debug(f'model inference on texts: {len(texts)}')
        batch = to_batch(texts, batch_size=batch_size)
        with torch.no_grad():
            output_text_embedding = []
            for i in batch:
                output_text_embedding += self.text_model.encode(i).tolist()

        logging.debug('compute similarity')
        # text size x image size
        sim = [[cosine_similarity(i, t) for i in output_image_embedding] for t in output_text_embedding]
        return sim

