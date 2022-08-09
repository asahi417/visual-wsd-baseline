import os
import json
from glob import glob
from os.path import join as pj

import pandas as pd

for _file in glob(pj('result', "*", "result.json")):
    with open(_file) as f:
        result = pd.DataFrame([json.loads(i) for i in f.read().split('\n')])
    # 'data': n,
    # 'gold': os.path.basename(d['Gold image']),
    # 'candidate': ranked_candidate,
    # 'prompt': '<>',
    # 'input_type': input_type