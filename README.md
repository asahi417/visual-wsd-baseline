# Visual Word Sense Disambiguation (V-WSD): Benchmark and Evaluation Script
This repository contains a baseline to solve Visual Word Sense Disambiguation (V-WSD) and the script to evaluate the results for the V-WSD.

## Get Started
```shell
git clone https://github.com/asahi417/vwsd_experiment
cd vwsd_experiment
pip install .
```

## Baseline with CLIP
As a baseline to solve V-WSD, we compute the cosine similarity of each candidate image, and the target phrase (or description) 
with [CLIP](https://arxiv.org/abs/2103.00020), and consider the image with the highest similarity as the prediction. This baseline can be 
obtained by following `vwsd-clip-baseline` command.
```shell
vwsd-clip-baseline [-h] [-d DATA_DIR] [-a ANNOTATION_FILE] [-m MODEL_CLIP] [-e EXPORT_DIR] [-p PROMPT [PROMPT ...]] [-b BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data-dir DATA_DIR
                        directly of images
  -a ANNOTATION_FILE, --annotation-file ANNOTATION_FILE
                        annotation file
  -m MODEL_CLIP, --model-clip MODEL_CLIP
                        clip model
  -e EXPORT_DIR, --export-dir EXPORT_DIR
                        export directly
  -p PROMPT [PROMPT ...], --prompt PROMPT [PROMPT ...]
                        prompt to be used in text embedding (specify the placeholder by <>)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size 
```

For example, baselines over CLIP model available on huggingface at the moment can be obtained by running commands below.
```shell
vwsd-clip-baseline --prompt 'This is <>.' 'Example of an image caption that explains <>.' '<>' -m 'openai/clip-vit-base-patch16' -e 'result/clip_vit_base_patch16'
vwsd-clip-baseline --prompt 'This is <>.' 'Example of an image caption that explains <>.' '<>' -m 'openai/clip-vit-base-patch16' -e 'result/clip_vit_base_patch16'
vwsd-clip-baseline --prompt 'This is <>.' 'Example of an image caption that explains <>.' '<>' -m 'openai/clip-vit-large-patch14' -e 'result/clip_vit_large_patch14'
vwsd-clip-baseline --prompt 'This is <>.' 'Example of an image caption that explains <>.' '<>' -m 'openai/clip-vit-large-patch14-336' -e 'result/clip_vit_large_patch14_336'  
```

## Evaluation
For each query (target word/full phrase/description) and candidate images, model will assign relevancy scores for all the candidates, which can be evaluated by ranking metrics.
To compute the ranking metrics, run following `vwsd-ranking-metric` command.

```shell
vwsd-ranking-metric [-h] [-r RANKING_FILES [RANKING_FILES ...]] [-m METRICS [METRICS ...]] [-e EXPORT]

compute ranking metrics

optional arguments:
  -h, --help            show this help message and exit
  -r RANKING_FILES [RANKING_FILES ...], --ranking-files RANKING_FILES [RANKING_FILES ...]
                        directly of model prediction
  -m METRICS [METRICS ...], --metrics METRICS [METRICS ...]
                        metrics to report (see https://amenra.github.io/ranx/metrics/)
  -e EXPORT, --export EXPORT
                        export file
```
Model prediction file should be a list of dictionary including 
```shell
{
  "data": 0,
  "gold": "image.172.jpg",
  "candidate": ["image.173.jpg", "image.180.jpg", "image.176.jpg", "image.172.jpg", "image.178.jpg", "image.181.jpg", "image.174.jpg", "image.175.jpg", "image.177.jpg", "image.179.jpg"],
  "relevance": [0.34015846252441406, 0.24420318603515626, 0.23036952972412109, 0.21879930496215821, 0.21160614013671875, 0.2113100242614746, 0.20634710311889648, 0.20474609375, 0.18968523025512696, 0.18504663467407226],
  "prompt": "This is <>.",
  "input_type": "Target word"
}
```

For example, ranking metric of the CLIP baseline can be obtained by running commands below.
```shell
vwsd-ranking-metric -r 'result/*/result.json' -m "map@5" "mrr@5" "ndcg@5" "map@10" "mrr@10" "ndcg@10" -e clip_baseline_result.csv
```

## CLIP Baseline Ranking Metrics

| map@5 | mrr@5 | ndcg@5 | map@10 | mrr@10 | ndcg@10 | prompt                                        | input_type  | CLIP                       |
|-------|-------|--------|--------|--------|---------|-----------------------------------------------|-------------|----------------------------|
|  76.7 |  76.7 |   81.3 |   77.4 |   77.4 |    82.9 | <>                                            | Definition  | clip_vit_base_patch16      |
|  77.7 |  77.7 |   82.1 |   78.5 |   78.5 |    83.8 | <>                                            | Full phrase | clip_vit_base_patch16      |
|  73.3 |  73.3 |   77.5 |   74.9 |   74.9 |    80.9 | Example of an image caption that explains <>. | Full phrase | clip_vit_base_patch16      |
|  73.9 |  73.9 |   79.2 |   74.8 |   74.8 |    81.0 | This is <>.                                   | Full phrase | clip_vit_base_patch16      |
|  67.3 |  67.3 |   72.9 |   68.4 |   68.4 |    76.0 | <>                                            | Target word | clip_vit_base_patch16      |
|  58.3 |  58.3 |   63.8 |   61.2 |   61.2 |    70.4 | Example of an image caption that explains <>. | Target word | clip_vit_base_patch16      |
|  66.3 |  66.3 |   72.1 |   67.6 |   67.6 |    75.3 | This is <>.                                   | Target word | clip_vit_base_patch16      |
|  78.5 |  78.5 |   83.9 |   78.5 |   78.5 |    83.9 | <>                                            | Definition  | clip_vit_base_patch32      |
|  75.4 |  75.4 |   80.4 |   76.3 |   76.3 |    82.2 | <>                                            | Full phrase | clip_vit_base_patch32      |
|  71.0 |  71.0 |   76.9 |   71.6 |   71.6 |    78.4 | Example of an image caption that explains <>. | Full phrase | clip_vit_base_patch32      |
|  67.1 |  67.1 |   75.4 |   67.1 |   67.1 |    75.4 | This is <>.                                   | Full phrase | clip_vit_base_patch32      |
|  66.4 |  66.4 |   71.1 |   68.8 |   68.8 |    76.3 | <>                                            | Target word | clip_vit_base_patch32      |
|  64.8 |  64.8 |   68.6 |   67.1 |   67.1 |    74.7 | Example of an image caption that explains <>. | Target word | clip_vit_base_patch32      |
|  55.0 |  55.0 |   60.1 |   58.7 |   58.7 |    68.5 | This is <>.                                   | Target word | clip_vit_base_patch32      |
|  78.3 |  78.3 |   82.6 |   78.8 |   78.8 |    84.1 | <>                                            | Definition  | clip_vit_large_patch14     |
|  73.3 |  73.3 |   77.6 |   74.6 |   74.6 |    80.8 | <>                                            | Full phrase | clip_vit_large_patch14     |
|  66.8 |  66.8 |   72.6 |   68.2 |   68.2 |    75.8 | Example of an image caption that explains <>. | Full phrase | clip_vit_large_patch14     |
|  74.6 |  74.6 |   78.5 |   76.0 |   76.0 |    81.8 | This is <>.                                   | Full phrase | clip_vit_large_patch14     |
|  66.2 |  66.2 |   69.5 |   68.9 |   68.9 |    76.0 | <>                                            | Target word | clip_vit_large_patch14     |
|  52.3 |  52.3 |   57.9 |   55.4 |   55.4 |    65.8 | Example of an image caption that explains <>. | Target word | clip_vit_large_patch14     |
|  57.4 |  57.4 |   63.0 |   60.1 |   60.1 |    69.5 | This is <>.                                   | Target word | clip_vit_large_patch14     |
|  75.8 |  75.8 |   80.8 |   76.3 |   76.3 |    82.2 | <>                                            | Definition  | clip_vit_large_patch14_336 |
|  74.3 |  74.3 |   79.6 |   75.0 |   75.0 |    81.2 | <>                                            | Full phrase | clip_vit_large_patch14_336 |
|  67.7 |  67.7 |   74.4 |   68.5 |   68.5 |    76.1 | Example of an image caption that explains <>. | Full phrase | clip_vit_large_patch14_336 |
|  77.1 |  77.1 |   80.3 |   78.5 |   78.5 |    83.6 | This is <>.                                   | Full phrase | clip_vit_large_patch14_336 |
|  63.7 |  63.7 |   66.6 |   67.4 |   67.4 |    75.0 | <>                                            | Target word | clip_vit_large_patch14_336 |
|  48.5 |  48.5 |   55.1 |   51.9 |   51.9 |    63.2 | Example of an image caption that explains <>. | Target word | clip_vit_large_patch14_336 |
|  57.6 |  57.6 |   64.3 |   59.7 |   59.7 |    69.2 | This is <>.                                   | Target word | clip_vit_large_patch14_336 |