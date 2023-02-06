""" Baseline of solving V-WSD with CLIP """
import argparse
import logging
import os
from os.path import join as pj

import pandas as pd
from vwsd import CLIP, MultilingualCLIP, data_loader, plot

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description="Solve V-WSD")
    parser.add_argument('-d', '--data-dir', help='directly of images', default='dataset', type=str)
    parser.add_argument('-l', '--language', help='language', default='en', type=str)
    parser.add_argument('-m', '--model-clip', help='clip model', default=None, type=str)
    parser.add_argument('-o', '--output-dir', help='output directly', default="result", type=str)
    parser.add_argument('-p', '--prompt', help='prompt to be used in text embedding (specify the placeholder by <>)',
                        type=str, nargs='+',
                        default=['<>', 'This is <>.', 'Example of an image caption that explains <>.'])
    parser.add_argument('--input-type', help='input text type',
                        type=str, nargs='+', default=['target word', 'target phrase'])
    parser.add_argument('-b', '--batch-size', help='batch size', default=None, type=int)
    parser.add_argument('--plot', help='', action='store_true')
    opt = parser.parse_args()

    # sanity check
    assert all("<>" in p for p in opt.prompt), "prompt need to contain `<>`"
    # os.makedirs(opt.output_dir, exist_ok=True)

    # load dataset
    data = data_loader(opt.data_dir)[opt.language]

    # load model
    if opt.language == 'en':
        clip = CLIP(opt.model_clip if opt.model_clip is not None else 'openai/clip-vit-large-patch14-336')
    else:
        clip = MultilingualCLIP(
            opt.model_clip if opt.model_clip is not None else 'sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # run inference
    result = []
    for n, d in enumerate(data):
        logging.info(f"{n+1}/{len(data)}: {d['target phrase']}")
        # output = []
        prompt_list = []
        for input_type in opt.input_type:
            prompt_list += [(p.replace("<>", d[input_type]), input_type, p) for p in opt.prompt]

        sim = clip.get_similarity(texts=[p[0] for p in prompt_list], images=d['candidate images'], batch_size=opt.batch_size)
        if opt.plot:
            plot(
                similarity=sim,
                texts=[p[0] for p in prompt_list],
                images=d['candidate images'],
                export_file=pj(opt.output_dir, "visualization", opt.language, f'similarity.{n}.png')
            )
        for (text, input_type, prompt_type), s in zip(prompt_list, sim):
            tmp = sorted(zip(s, d['candidate images']), key=lambda x: x[0], reverse=True)
            result.append({
                'language': opt.language,
                'data': n,
                'candidate': [os.path.basename(i[1]) for i in tmp],
                'relevance': sorted(s, reverse=True),
                'text': text,
                'input_type': input_type,
                'prompt': prompt_type
            })

    df = pd.DataFrame(result)
    for (prompt, input_type), g in df.groupby(['prompt', 'input_type']):
        path = pj(opt.output_dir, f'{prompt.replace("<>", "mask")}.{input_type}'.replace(" ", "_"))
        os.makedirs(path, exist_ok=True)
        with open(pj(path, f'prediction.{opt.language}.txt'), 'w') as f:
            f.write('\n'.join(['\t'.join(x) for x in g.sort_values(by=['data'])['candidate'].to_list()]))
        g.to_csv(pj(path, f'full_result.{opt.language}.csv'), index=False)


if __name__ == '__main__':
    main()
