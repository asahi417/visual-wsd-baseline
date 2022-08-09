import os
import requests
import tarfile
import zipfile
import gzip
from os.path import join as pj
import pandas as pd


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
    if not os.path.exists(path_to_dataset):
        wget('https://github.com/asahi417/vwsd_experiment/releases/download/dataset/dataset.zip',
             cache_dir=os.path.dirname(path_to_dataset))

    path_to_image_dir = pj(path_to_dataset, 'images')
    path_to_annotation_file = pj(path_to_dataset, 'annotations.csv')
    assert os.path.exists(path_to_annotation_file), f'annotation not found at {path_to_annotation_file}'
    assert os.path.isdir(path_to_image_dir), f'images not found at {path_to_image_dir}'
    annotation = pd.read_csv(path_to_annotation_file)
    candidate_columns = [i for i in annotation.columns if i.startswith('Candidate')]
    annotation = list(annotation.T.to_dict().values())
    dataset = []
    for single_ann in annotation:
        gold_image = pj(path_to_image_dir, single_ann['Full phrase'], single_ann['Gold image'])
        assert os.path.exists(gold_image), f"missing gold image {gold_image}"
        candidate_images = [pj(path_to_image_dir, single_ann['Full phrase'], single_ann[c]) for c in candidate_columns]
        assert all(os.path.exists(i) for i in candidate_images), \
            f"missing image ({[i for i in candidate_images if not os.path.exists(i)]})"
        assert gold_image in candidate_images, f"gold image {gold_image} not in the candidates {candidate_images}"
        dataset.append({
            "Target word": single_ann['Target word'],
            "Full phrase": single_ann['Full phrase'],
            "Definition": single_ann['Definition'],
            "Gold image": gold_image,
            "Candidate images": candidate_images
        })
    return dataset
