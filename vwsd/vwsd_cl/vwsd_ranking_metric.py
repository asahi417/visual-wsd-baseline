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
    parser.add_argument('-r', '--ranking-files', help='directly of model predictions', type=str, nargs='+',
                        default=[pj('result', "*", "result.json")])
    parser.add_argument('-m', '--metrics', help='metrics to report (see https://amenra.github.io/ranx/metrics/)',
                        type=str, nargs='+',
                        default=["hit_rate@1", "map@5", "mrr@5", "ndcg@5", "map@10", "mrr@10", "ndcg@10"])
    parser.add_argument('-e', '--export', help='export file', default='rank_metrics.csv', type=str)

    # parser.add_argument('--return-ci', action='store_true', help='return confidence interval by bootstrap')
    opt = parser.parse_args()
    ranking_files = [opt.ranking_files] if type(opt.ranking_files) is str else opt.ranking_files
    ranking_files = list(chain(*[glob(i) for i in ranking_files]))
    logging.info(f'{len(ranking_files)} files')

    metrics = []
    for _file in ranking_files:
        with open(_file) as f:
            result = pd.DataFrame([json.loads(i) for i in f.read().split('\n')])
            for (prompt, input_type), df in result.groupby(by=['prompt', 'input_type']):
                qrels_dict = {str(_df['data']): {_df['gold']: 1} for _, _df in df.iterrows()}
                run_dict = {str(_df['data']): {c: r for c, r in zip(_df['candidate'], _df['relevance'])}
                            for _, _df in df.iterrows()}
                metric = evaluate(Qrels(qrels_dict), Run(run_dict), metrics=opt.metrics)
                metric = {k: 100 * v for k, v in metric.items()}
                metric.update({'prompt': prompt, 'input_type': input_type, 'file': _file})
                metrics.append(metric)
    df = pd.DataFrame(metrics)
    df.to_csv(opt.export, index=False)
    for input_type, g in df.groupby("input_type"):
        logging.info(input_type)
        g.pop("input_type")
        g['model'] = [os.path.basename(os.path.dirname(i)) for i in g.pop('file')]
        logging.info("\n" + g.round(1).to_markdown(index=False))


if __name__ == '__main__':
    main()
