import argparse
import json
import logging
import os

from vwsd import CLIP, data_loader

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description="Solve V-WSD")
    parser.add_argument('-i', '--image-dir', help='directly of images', default='dataset/images', type=str)
    parser.add_argument('-a', '--annotation-file', help='annotation file', default='dataset/annotations.csv', type=str)
    parser.add_argument('-m', '--model-clip', help='clip model', default='openai/clip-vit-base-patch32', type=str)
    parser.add_argument('-b', '--batch-size', help='batch size', default=32, type=int)
    # parser.add_argument('--return-ci', action='store_true', help='return confidence interval by bootstrap')
    opt = parser.parse_args()

    data = data_loader(path_to_image_dir=opt.image_dir, path_to_annotation_file=opt.annotation_file)
    clip = CLIP(opt.model_clip)