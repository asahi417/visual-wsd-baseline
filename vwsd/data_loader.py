import os
import requests
import tarfile
import zipfile
import gzip
from itertools import chain
from os.path import join as pj

url_image = 'https://github.com/asahi417/visual-wsd-baseline/releases/download/dataset-test/test_images_resized.zip'
url_label = "https://github.com/asahi417/visual-wsd-baseline/releases/download/dataset-test/test.data.v1.1.zip"


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


def data_loader(cache_dir: str = 'dataset'):
    path_image = pj(cache_dir, "image")
    path_label = pj(cache_dir, "label")
    if not os.path.exists(path_image):
        wget(url_image, cache_dir=path_image)
    if not os.path.exists(path_label):
        wget(url_label, cache_dir=path_label)
    dir_image = pj(cache_dir, "image", "test_images_resized")

    def _load(_file):
        with open(_file) as _f:
            tmp = [x.split('\t') for x in _f.read().split('\n') if len(x) > 0]
        return [{"target word": x[0], "target phrase": x[1], "candidate images": [pj(dir_image, y) for y in x[2:]]} for x in tmp]

    data = {i.split('.')[0]: _load(pj(cache_dir, "label", i))
            for i in ["en.test.data.v1.1.txt", "fa.test.data.txt", "it.test.data.v1.1.txt"]}
    # validate image path
    image_paths = list(chain(*[list(chain(*[i['candidate images'] for i in d])) for d in data.values()]))
    if not all([os.path.exists(i) for i in image_paths]):
        raise ValueError(f"Image path is invalid. Please check the path: "
                         f"{[i for i in image_paths if not os.path.exists(i)]}")

    return data
