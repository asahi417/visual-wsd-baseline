import os
import json
from glob import glob
from os.path import join as pj

import pandas as pd
from ranx import Qrels, Run, evaluate


for _file in glob(pj('result', "*", "result.json")):
    with open(_file) as f:
        result = pd.DataFrame([json.loads(i) for i in f.read().split('\n')])
        for tag, df in result.groupby(by=['prompt', 'input_type']):
            qrels_dict = {str(_df['data']): {_df['gold']: 1} for _, _df in df.iterrows()}
            run_dict = {str(_df['data']): {c: r for c, r in zip(_df['candidate'], _df['relevance'])}
                        for _, _df in df.iterrows()}
            qrels = Qrels(qrels_dict)
            run = Run(run_dict)

    # 'data': n,
    # 'gold': os.path.basename(d['Gold image']),
    # 'candidate': ranked_candidate,
    # 'prompt': '<>',
    # 'input_type': input_type