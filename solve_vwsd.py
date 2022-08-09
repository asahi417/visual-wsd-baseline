""" Baseline of solving V-WSD with CLIP
python solve_vwsd.py -m 'openai/clip-vit-base-patch16' -e 'result/clip_vit_base_patch16'
python solve_vwsd.py -m 'openai/clip-vit-base-patch32' -e 'result/clip_vit_base_patch32'
python solve_vwsd.py -m 'openai/clip-vit-large-patch14' -e 'result/clip_vit_large_patch14'
python solve_vwsd.py -m 'openai/clip-vit-large-patch14-336' -e 'result/clip_vit_large_patch14_336'
"""
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
max_character = 40


def cap_text(_string):
    if len(_string) < max_character:
        return _string
    sentence = []
    new_string = []
    for word in _string.split(' '):
        new_string.append(word)
        if len(' '.join(new_string)) > max_character:
            sentence.append(' '.join(new_string))
            new_string = []
    if len(new_string) != 0:
        sentence.append(' '.join(new_string))
    return '\n'.join(sentence)


def plot(similarity, texts, images, export_file):
    assert similarity.shape[0] == len(texts) and similarity.shape[1] == len(images), \
        f"{similarity.shape} != {(len(images), len(texts))}"
    plt.figure(figsize=(22, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(len(texts)), [cap_text(i) for i in texts], fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(Image.open(image).convert("RGB"), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    for x in range(len(images)):
        for y in range(len(texts)):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    plt.xlim([-0.5, len(images) - 0.5])
    plt.ylim([len(texts) + 0.5, -2])
    plt.tight_layout()
    plt.savefig(export_file, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description="Solve V-WSD")
    parser.add_argument('-d', '--data-dir', help='directly of images', default='dataset', type=str)
    parser.add_argument('-a', '--annotation-file', help='annotation file', default='dataset/annotations.csv', type=str)
    parser.add_argument('-m', '--model-clip', help='clip model', default='openai/clip-vit-base-patch32', type=str)
    parser.add_argument('-e', '--export-dir', help='export directly', default='result', type=str)
    parser.add_argument('-p', '--prompt', help='prompt to be used in text embedding (specify the placeholder by {})',
                        type=str, nargs='+',
                        default=['This is <>.', 'Example of an image caption that explains <>.', '<>'])
    parser.add_argument('-b', '--batch-size', help='batch size', default=None, type=int)
    # parser.add_argument('--return-ci', action='store_true', help='return confidence interval by bootstrap')
    opt = parser.parse_args()

    os.makedirs(opt.export_dir, exist_ok=True)
    data = data_loader(opt.data_dir)
    clip = CLIP(opt.model_clip)
    result = []
    for n, d in enumerate(data):
        logging.info(f'PROGRESS: {n + 1}/{len(data)}')
        output = []
        for input_type in ['Target word', 'Full phrase', 'Definition']:
            if input_type != 'Definition':
                for p in tqdm(opt.prompt):
                    assert '<>' in p, f'prompt needs `<>` to specify placeholder: {p}'
                    text = p.replace("<>", d[input_type])
                    _, _, sim = clip.get_embedding(
                        texts=[text], images=d['Candidate images'], return_similarity=True, batch_size=opt.batch_size
                    )
                    output.append((sim * 0.01, text, input_type.split(' ')[0]))
                    ranked_candidate = [os.path.basename(i[1]) for i in
                                        sorted(zip(sim, d['Candidate images']), key=lambda x: x[0], reverse=True)]
                    result.append({
                        'data': n,
                        'gold': os.path.basename(d['Gold image']),
                        'candidate': ranked_candidate,
                        'prompt': p,
                        'input_type': input_type
                    })
            else:
                text = d[input_type]
                _, _, sim = clip.get_embedding(
                    texts=[text], images=d['Candidate images'], return_similarity=True, batch_size=opt.batch_size
                )
                output.append((sim * 0.01, text, input_type.split(' ')[0]))
                ranked_candidate = [os.path.basename(i[1]) for i in
                                    sorted(zip(sim, d['Candidate images']), key=lambda x: x[0], reverse=True)]
                result.append({
                    'data': n,
                    'gold': os.path.basename(d['Gold image']),
                    'candidate': ranked_candidate,
                    'prompt': '<>',
                    'input_type': input_type
                })
        d['Candidate images'].pop(d['Candidate images'].index(d['Gold image']))

        plot(
            similarity=np.concatenate([i[0] for i in output], 1).T,
            texts=[f"`{i[1]}` [{i[2]}]" for i in output],
            images=[d['Gold image']] + d['Candidate images'],
            export_file=pj(opt.export_dir, f'similarity.{n}.png')
        )

    with open(pj(opt.export_dir, 'result.json'), 'w') as f:
        f.write('\n'.join([json.dumps(i) for i in result]))


if __name__ == '__main__':
    main()
