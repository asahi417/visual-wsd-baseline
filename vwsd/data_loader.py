import os
from os.path import join as pj
import pandas as pd


def data_loader(path_to_image_dir: str = pj('dataset', 'images'),
                path_to_annotation_file: str = pj('dataset', 'annotations.csv')):
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


if __name__ == '__main__':
    print(data_loader())
