from vwsd import CLIP

models = [
    'openai/clip-vit-large-patch14',
    'openai/clip-vit-base-patch32',
    'openai/clip-vit-large-patch14-336',
    'openai/clip-vit-base-patch16'
]
for m in models:
    CLIP(m)
