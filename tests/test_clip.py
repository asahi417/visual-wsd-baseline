import os
import logging
from os.path import join as pj
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR100
from PIL import Image

from vwsd import CLIP

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
model = CLIP('openai/clip-vit-base-patch32')
image_dir = 'test_images'
export_dir = 'test_outputs'
os.makedirs(export_dir, exist_ok=True)


######################
# get image & encode #
######################
descriptions = {
    "page.png": "a page of text about segmentation",
    "chelsea.png": "a facial photo of a tabby cat",
    "astronaut.png": "a portrait of an astronaut with the American flag",
    "rocket.jpg": "a rocket standing on a launchpad",
    "motorcycle_right.png": "a red motorcycle standing in a garage",
    "camera.png": "a person looking at a camera on a tripod",
    "horse.png": "a black-and-white silhouette of a horse",
    "coffee.png": "a cup of coffee on a saucer"
}
texts = []
images = []
plt.figure(figsize=(16, 5))
for n, (k, v) in enumerate(descriptions.items()):
    images.append(pj(image_dir, k))
    texts.append(f"This is {v}")
    plt.subplot(2, 4, n + 1)
    plt.imshow(Image.open(pj(image_dir, k)).convert("RGB"))
    plt.title(f"{k}\n{v}")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig(pj(export_dir, 'test_images.png'))

##############
# similarity #
##############
logging.info('** Text-Image similarity ** ')
_, _, sim = model.get_embedding(images=images, texts=texts, return_similarity=True)
similarity = sim.T * 0.01

# plot
plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
plt.yticks(range(len(descriptions)), texts, fontsize=18)
plt.xticks([])
for i, image in enumerate(images):
    plt.imshow(Image.open(image).convert("RGB"), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)
plt.xlim([-0.5, len(descriptions) - 0.5])
plt.ylim([len(descriptions) + 0.5, -2])
plt.title("Cosine similarity between text and image features", size=20)
plt.tight_layout()
plt.savefig(pj(export_dir, 'similarity.png'))

###########################
# Zeroshot classification #
###########################
logging.info('** ZERO-SHOT ** ')
# encode text representing each class
cifar100_class = CIFAR100("cache", download=True).classes
texts = [f"This is a photo of a {label}" for label in cifar100_class]
_, _, sim = model.get_embedding(images=images, texts=texts, return_similarity=True, return_tensor=True)
prob = sim.softmax(dim=1)  # we can take the softmax to get the label probabilities
top_probs, top_labels = prob.cpu().topk(5, dim=-1)

# plot
plt.figure(figsize=(16, 16))
for i, image in enumerate(images):
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(Image.open(image).convert("RGB"))
    plt.axis("off")
    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [cifar100_class[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")
plt.subplots_adjust(wspace=0.5)
plt.tight_layout()
plt.savefig(pj(export_dir, 'zeroshot.png'))

