import os.path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def cap_text(_string, max_character: int = 40):
    if len(_string) < max_character:
        return _string
    sentence = []
    new_string = []
    for word in _string.split(' '):
        new_string.append(word)
        if len(' '.join(new_string)) > max_character:
            sentence.append(' '.join(new_string))
            new_string = []
    if len(new_string) != 0:
        sentence.append(' '.join(new_string))
    return '\n'.join(sentence)


def plot(similarity, texts, images, export_file, gold_image_index=None):
    similarity = np.array(similarity) * 0.01
    assert similarity.shape[0] == len(texts) and similarity.shape[1] == len(images), \
        f"{similarity.shape} != {(len(images), len(texts))}"
    plt.figure(figsize=(22, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(len(texts)), [cap_text(i) for i in texts], fontsize=18)
    if gold_image_index is not None:
        plt.xticks(range(len(images)), ['' if i != gold_image_index else 'True Image' for i in range(len(images))],
                   fontsize=18)
    else:
        plt.xticks(range(len(images)), fontsize=18)
    for i, image in enumerate(images):
        plt.imshow(Image.open(image).convert("RGB"), extent=(i - 0.5, i + 0.5, -2.0, -1), origin="lower")

    for x in range(len(images)):
        for y in range(len(texts)):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    plt.gca().xaxis.tick_top()
    plt.xlim([-0.5, len(images) - 0.5])
    plt.ylim([len(texts) + 0.5, -2])
    plt.tight_layout()
    if os.path.dirname(export_file) != "":
        os.makedirs(os.path.dirname(export_file), exist_ok=True)
    plt.savefig(export_file, bbox_inches='tight')
