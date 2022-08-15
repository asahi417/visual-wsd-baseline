""" Baseline of solving V-WSD with CLIP """
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
max_character = 40  # for plot


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


def plot(similarity, texts, images, gold_image_index, export_file):
    assert similarity.shape[0] == len(texts) and similarity.shape[1] == len(images), \
        f"{similarity.shape} != {(len(images), len(texts))}"
    plt.figure(figsize=(22, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(len(texts)), [cap_text(i) for i in texts], fontsize=18)
    plt.xticks(range(len(images)), ['' if i != gold_image_index else 'True Image' for i in range(len(images))],
               fontsize=18)
    for i, image in enumerate(images):
        plt.imshow(Image.open(image).convert("RGB"), extent=(i - 0.5, i + 0.5, -2.0, -1), origin="lower")

    for x in range(len(images)):
        for y in range(len(texts)):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    plt.gca().xaxis.tick_top()
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
    parser.add_argument('-p', '--prompt', help='prompt to be used in text embedding (specify the placeholder by <>)',
                        type=str, nargs='+',
                        default=['This is <>.', 'Example of an image caption that explains <>.'])
    parser.add_argument('--input-type', help='input text type',
                        type=str, nargs='+', default=['Target word', 'Full phrase'])
    parser.add_argument('-b', '--batch-size', help='batch size', default=None, type=int)
    parser.add_argument('--skip-default-prompt', help='skip testing preset prompts', action='store_true')
    opt = parser.parse_args()
    assert all("<>" in p for p in opt.prompt), "prompt need to contain `<>`"
    os.makedirs(opt.export_dir, exist_ok=True)
    data = data_loader(opt.data_dir)
    clip = CLIP(opt.model_clip)
    result = []
    for n, d in enumerate(data):
        logging.info(f'PROGRESS: {n + 1}/{len(data)}')
        output = []
        prompt_list = []
        if not opt.skip_default_prompt:
            prompt_list += [
                (d['Definition'], 'Definition'),
                (d['Target word'], 'Target word'),
                (d['Full phrase'], 'Full phrase'),
                (f"{d['Target word']}:{d['Definition']}", 'Definition, Target word'),
                (f"{d['Full phrase']}:{d['Definition']}", 'Definition, Full phrase')
            ]
        for input_type in opt.input_type:
            prompt_list += [(p.replace("<>", d[input_type]), input_type) for p in opt.prompt]

        for text, input_type in tqdm(prompt_list):
            _, _, sim = clip.get_embedding(
                texts=[text], images=d['Candidate images'], return_similarity=True, batch_size=opt.batch_size
            )
            output.append((sim * 0.01, text))
            tmp = sorted(zip(sim, d['Candidate images']), key=lambda x: x[0], reverse=True)
            ranked_candidate = [os.path.basename(i[1]) for i in tmp]
            relevance = [i[0].tolist()[0] * 0.01 for i in tmp]
            result.append({
                'data': n,
                'gold': os.path.basename(d['Gold image']),
                'candidate': ranked_candidate,
                'relevance': relevance,
                'prompt': text,
                'input_type': input_type
            })
        gold_image_index = d['Candidate images'].index(d['Gold image'])

        plot(
            similarity=np.concatenate([i[0] for i in output], 1).T,
            texts=[i[1] for i in output],
            images=d['Candidate images'],
            gold_image_index=gold_image_index,
            export_file=pj(opt.export_dir, f'similarity.{n}.png')
        )

    with open(pj(opt.export_dir, 'result.json'), 'w') as f:
        f.write('\n'.join([json.dumps(i) for i in result]))


if __name__ == '__main__':
    main()
