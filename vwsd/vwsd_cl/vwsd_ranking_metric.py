import argparse
import logging
import os
import json
from os.path import join as pj
from statistics import mean
import pandas as pd
from ranx import Qrels, Run, evaluate


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_metric(prediction, gold_ranking):
    """ Get MRR and HIT Rate

    >>> p = ['a', 'b', 'c']
    >>> r = [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd']]
    >>> mrr_score, hit_score = get_metric(p, r)
    >>> mrr_score
    0.611
    >>> hit_score
    0.333
    """
    assert all(len(set(p)) == len(p) for p in prediction), prediction
    assert all(r in p for p, r in zip(prediction, gold_ranking)), list(zip(prediction, gold_ranking))
    mrr = mean(1/(p.index(r) + 1) for p, r in zip(prediction, gold_ranking))
    hit = mean(p.index(r) == 0 for p, r in zip(prediction, gold_ranking))
    return mrr, hit



def main():
    parser = argparse.ArgumentParser(description="compute ranking metrics")
    parser.add_argument('-r', '--reference', help='reference label file', type=str, required=True)
    parser.add_argument('-p', '--prediction-dir', help='', type=str, required=True)
    parser.add_argument('-m', '--metrics', help='metrics to report (see https://amenra.github.io/ranx/metrics/)',
                        type=str, nargs='+',
                        default=["hit_rate@1", "map@5", "mrr@5", "ndcg@5", "map@10", "mrr@10", "ndcg@10"])
    parser.add_argument('-o', '--output-file', help='export file', default='rank_metrics.jsonl', type=str)
    opt = parser.parse_args()

    # open reference file
    with open(opt.reference, 'r') as f:
        tmp = [i.split('\t') for i in f.read().split("\n") if len(i) > 0]
    reference = {i: [x for x, y in tmp if y == i] for i in ['en', 'fa', 'it']}

    # open prediction
    def safe_open(path):
        with open(path) as f_read:
            return [i.split('\t') for i in f_read.read().split("\n") if len(i) > 0]
    prediction = {i: safe_open(pj(opt.prediction_dir, f"prediction.{i}.txt")) for i in ['en', 'fa', 'it']}

    # compute metrics
    metric_dict = {'model': opt.prediction_dir}
    for i in ['en', 'fa', 'it']:
        qrels_dict = {str(n): {r: 1} for n, r in enumerate(reference[i])}
        run_dict = {str(n): {c: 1/(1 + r) for r, c in enumerate(x)} for n, x in enumerate(prediction[i])}
        m, h = get_metric(prediction[i], reference[i])
        metric = {f'mrr_official/{i}': m, f'hit_official/{i}': h}
        metric.update({f"{k}/{i}": v for k, v in evaluate(Qrels(qrels_dict), Run(run_dict), metrics=opt.metrics).items()})
        metric_dict.update(metric)
    for m in ['mrr_official', 'hit_official'] + opt.metrics:
        metric_dict[f"{m}/avg"] = sum([metric_dict[f"{m}/{i}"] for i in ['en', 'fa', 'it']]) / 3

    if os.path.exists(opt.output_file):
        with open(opt.output_file) as f:
            metric_all = [json.loads(i) for i in f.read().split("\n") if len(i) > 0]
    else:
        metric_all = []
    metric_all.append(metric_dict)
    with open(opt.output_file, 'w') as f:
        f.write("\n".join([json.dumps(i) for i in metric_all]))
    print(pd.DataFrame(metric_all).to_markdown(index=False))


if __name__ == '__main__':
    main()
