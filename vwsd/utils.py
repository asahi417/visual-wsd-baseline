import tarfile
import zipfile
import gzip
import os
import random
import requests

import numpy as np
import torch

default_cache_dir = f"{os.path.expanduser('~')}/.cache/vwsd"


def fix_seed(seed: int = 12):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def wget(url, cache_dir: str):
    """ wget and uncompress """
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    path = '{}/{}'.format(cache_dir, filename)
    if not os.path.exists(path):
        with open('{}/{}'.format(cache_dir, filename), "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    if path.endswith('.tar.gz') or path.endswith('.tgz') or path.endswith('.tar'):
        if path.endswith('.tar'):
            tar = tarfile.open(path)
        else:
            tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
        os.remove(path)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            with open(path.replace('.gz', ''), 'wb') as f_write:
                f_write.write(f.read())
        os.remove(path)


