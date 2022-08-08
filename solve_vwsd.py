import argparse
import json
import logging
import os
from os.path import join as pj

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from vwsd import CLIP, data_loader

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def plot():
    # plot
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(len(descriptions)), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(Image.open(image).convert("RGB"), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    plt.xlim([-0.5, len(descriptions) - 0.5])
    plt.ylim([len(descriptions) + 0.5, -2])
    plt.title("Cosine similarity between text and image features", size=20)
    plt.tight_layout()
    plt.savefig(pj(export_dir, 'similarity.png'))


def main():
    parser = argparse.ArgumentParser(description="Solve V-WSD")
    parser.add_argument('-d', '--data-dir', help='directly of images', default='dataset', type=str)
    parser.add_argument('-a', '--annotation-file', help='annotation file', default='dataset/annotations.csv', type=str)
    parser.add_argument('-m', '--model-clip', help='clip model', default='openai/clip-vit-base-patch32', type=str)
    parser.add_argument('-e', '--export-dir', help='export directly', default='result', type=str)
    # parser.add_argument('-i', '--input-type', help='input type', default='Full phrase', type=str)
    parser.add_argument('-p', '--prompt', help='prompt to be used in text embedding (specify the placeholder by {})',
                        type=str, nargs='+',
                        default=['This is <>.', 'Example of an image caption that explains <>.', '<>'])
    parser.add_argument('-b', '--batch-size', help='batch size', default=None, type=int)
    # parser.add_argument('--return-ci', action='store_true', help='return confidence interval by bootstrap')
    opt = parser.parse_args()
    # assert opt.input_type in ['Target word', 'Full phrase', 'Definition'], f'{opt.input_type} is invalid'

    os.makedirs(opt.data_dir, exist_ok=True)
    data = data_loader(opt.data_dir)
    clip = CLIP(opt.model_clip)
    accuracy = []
    for n, d in enumerate(data):
        logging.info(f'PROGRESS: {n + 1}/{len(data)}')
        output = []
        d['Gold image']
        for input_type in ['Target word', 'Full phrase', 'Definition']:
            if input_type != 'Definition':
                for p in tqdm(opt.prompt):
                    assert '<>' in p, f'prompt needs `<>` to specify placeholder: {p}'
                    text = p.replace("<>", d[input_type])
                    _, _, sim = clip.get_embedding(
                        texts=[text], images=d['Candidate images'], return_similarity=True, batch_size=opt.batch_size
                    )
                    output.append((sim * 0.01, text, input_type.split(' ')[0]))
                    ranked_candidate = sorted(zip(sim, d['Candidate images']), key=lambda x: x[0], reverse=True)
                    print(ranked_candidate)
                    input()
            else:
                text = d[input_type]
                _, _, sim = clip.get_embedding(
                    texts=[text], images=d['Candidate images'], return_similarity=True, batch_size=opt.batch_size
                )
                output.append((sim.cpu().numpy() * 0.01, text, input_type.split(' ')[0]))
        similarity = np.concatenate([i[0] for i in output], 1)
        texts = [f"{i[1]} ({i[2]})" for i in output]
        print(similarity.shape)
        input()

if __name__ == '__main__':
    main()