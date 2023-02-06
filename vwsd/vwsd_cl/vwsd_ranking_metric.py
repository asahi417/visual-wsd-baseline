import argparse
import logging
import json
import os
from itertools import chain
from glob import glob
from os.path import join as pj

import pandas as pd
from ranx import Qrels, Run, evaluate


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description="compute ranking metrics")
    parser.add_argument('-r', '--reference', help='reference label file', type=str, required=True)
    parser.add_argument('-p', '--prediction-dir', help='', type=str, required=True)
    parser.add_argument('-m', '--metrics', help='metrics to report (see https://amenra.github.io/ranx/metrics/)',
                        type=str, nargs='+',
                        default=["hit_rate@1", "map@5", "mrr@5", "ndcg@5", "map@10", "mrr@10", "ndcg@10"])
    parser.add_argument('-o', '--output-file', help='export file', default='rank_metrics.csv', type=str)
    opt = parser.parse_args()

    # open reference file
    with open(opt.reference, 'r') as f:
        tmp = [i.split('\t') for i in f.read().split("\n") if len(i) > 0]
    reference = {i: [x for x, y in tmp if y == i] for i in ['en', 'fa', 'it']}

    # open prediction
    def safe_open(path):
        with open(path) as f_read:
            return [i.split('\t') for i in f_read.read().split("\n") if len(i) > 0]
    prediction = {i: safe_open(i) for i in ['en', 'fa', 'it']}

    # compute metrics
    metrics = []
    for i in ['en', 'fa', 'it']:
        qrels_dict = {str(n): {r: 1} for n, r in enumerate(reference[i])}
        run_dict = {str(n): {c: r for c, r in zip(_df['candidate'], _df['relevance'])}
                    for n, _df in enumerate(prediction[i])}
        metric = evaluate(Qrels(qrels_dict), Run(run_dict), metrics=opt.metrics)
        metric = {k: 100 * v for k, v in metric.items()}
        metric.update({'language': i, 'prediction_dir': opt.prediction_dir})
        metrics.append(metric)
    df = pd.DataFrame(metrics)
    df.to_csv(opt.output_file, index=False)
    # for input_type, g in df.groupby("input_type"):
    #     logging.info(input_type)
    #     g.pop("input_type")
    #     g['model'] = [os.path.basename(os.path.dirname(i)) for i in g.pop('file')]
    #     logging.info("\n" + g.round(1).to_markdown(index=False))


if __name__ == '__main__':
    main()
