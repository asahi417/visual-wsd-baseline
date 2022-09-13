import os
import requests
import tarfile
import zipfile
import gzip
from os.path import join as pj


def wget(url, cache_dir: str = '.'):
    """ wget and uncompress """
    if cache_dir != '':
        os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    path = pj(cache_dir, filename)
    if not os.path.exists(path):
        with open(path, "wb") as f:
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


def data_loader(path_to_dataset: str = 'dataset'):
    # path_to_dataset = pj(path_to_dataset, "semeval-2023-task-1-V-WSD-trial-v1")
    if not os.path.exists(path_to_dataset):
        wget('https://github.com/asahi417/visual-wsd-baseline/releases/download/dataset-v2/semeval-2023-task-1-V-WSD-trial-v1.tar.gz',
             cache_dir=os.path.dirname(path_to_dataset))
    path_to_image_dir = pj(path_to_dataset, 'all_images')
    with open(pj(path_to_dataset, 'trial.data.txt')) as f:
        data = [i.split('\t') for i in f.read().split('\n') if len(i) > 0]

    with open(pj(path_to_dataset, 'trial.gold.txt')) as f:
        true_image = [i for i in f.read().split('\n') if len(i) > 0]

    assert len(data) == len(true_image), f"{len(data)} != {len(true_image)}"

    dataset = []
    for d, i in zip(data, true_image):
        candidate_images = [pj(path_to_image_dir, p) for p in d[2:]]
        assert all(os.path.exists(p) for p in candidate_images), candidate_images
        gold_image = pj(path_to_image_dir, i)
        assert os.path.exists(gold_image), gold_image
        dataset.append({
            "Target word": d[0],
            "Full phrase": d[1],
            "Gold image": gold_image,
            "Candidate images": candidate_images
        })
    return dataset
