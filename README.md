# Visual Word Sense Disambiguation (Visual-WSD): Benchmark and Evaluation Script
This repository contains a baseline to solve Visual Word Sense Disambiguation (V-WSD) and the script to evaluate the results for the V-WSD.

## Get Started
```shell
git clone https://github.com/asahi417/vwsd_experiment
cd vwsd_experiment
pip install .
```

## Baseline with CLIP


<p align="center">
  <img src="result/visualization/en/similarity.0.png">
</p>

As a baseline to solve V-WSD, we use [CLIP](https://arxiv.org/abs/2103.00020) to compute the text and image embeddings, 
and rank the candidate images based on the cosine similarity between the text and image embeddings.
Following command will run the baseline for each language. 
```shell
vwsd-clip-baseline -l en --plot
vwsd-clip-baseline -l fa --plot
vwsd-clip-baseline -l it --plot
```

## Evaluation
For each query (target word/full phrase) and candidate images, model will assign relevancy scores for all the candidates, which can be evaluated by ranking metrics.
To compute the ranking metrics, run following `vwsd-ranking-metric` command.

```shell
vwsd-ranking-metric -p result/Example_of_an_image_caption_that_explains_mask..target_phrase -r gold.txt
vwsd-ranking-metric -p result/Example_of_an_image_caption_that_explains_mask..target_word -r gold.txt
vwsd-ranking-metric -p result/mask.target_phrase -r gold.txt
vwsd-ranking-metric -p result/mask.target_word -r gold.txt
vwsd-ranking-metric -p result/This_is_mask..target_phrase -r gold.txt
vwsd-ranking-metric -p result/This_is_mask..target_word -r gold.txt
```

### Baseline Results

| model                                                                |   hit_rate@1/en |   map@5/en |   mrr@5/en |   ndcg@5/en |   map@10/en |   mrr@10/en |   ndcg@10/en |   hit_rate@1/fa |   map@5/fa |   mrr@5/fa |   ndcg@5/fa |   map@10/fa |   mrr@10/fa |   ndcg@10/fa |   hit_rate@1/it |   map@5/it |   mrr@5/it |   ndcg@5/it |   map@10/it |   mrr@10/it |   ndcg@10/it |   hit_rate@1/avg |   map@5/avg |   mrr@5/avg |   ndcg@5/avg |   map@10/avg |   mrr@10/avg |   ndcg@10/avg |
|:---------------------------------------------------------------------|----------------:|-----------:|-----------:|------------:|------------:|------------:|-------------:|----------------:|-----------:|-----------:|------------:|------------:|------------:|-------------:|----------------:|-----------:|-----------:|------------:|------------:|------------:|-------------:|-----------------:|------------:|------------:|-------------:|-------------:|-------------:|--------------:|
| result/Example_of_an_image_caption_that_explains_mask..target_phrase |         50.324  |    65.2592 |    65.2592 |     71.162  |     66.8163 |     66.8163 |      74.8529 |            18.5 |    34.2583 |    34.2583 |     41.9426 |     38.905  |     38.905  |      53.1431 |         19.0164 |    32.0492 |    32.0492 |     38.2175 |     37.596  |     37.596  |      51.9219 |          29.2801 |     43.8556 |     43.8556 |      50.4407 |      47.7724 |      47.7724 |       59.9726 |
| result/Example_of_an_image_caption_that_explains_mask..target_word   |         29.1577 |    44.964  |    44.964  |     52.1259 |     48.3614 |     48.3614 |      60.4849 |            16.5 |    31.4083 |    31.4083 |     38.9235 |     36.3502 |     36.3502 |      51.0824 |         13.4426 |    23.6448 |    23.6448 |     28.9363 |     30.5109 |     30.5109 |      46.1987 |          19.7001 |     33.339  |     33.339  |      39.9953 |      38.4075 |      38.4075 |       52.5887 |
| result/mask.target_phrase                                            |         60.4752 |    72.8582 |    72.8582 |     77.7656 |     73.8763 |     73.8763 |      80.2202 |            28.5 |    43.1917 |    43.1917 |     50.5394 |     46.6974 |     46.6974 |      59.1721 |         22.623  |    38.4809 |    38.4809 |     45.7448 |     42.6063 |     42.6063 |      55.9699 |          37.1994 |     51.5102 |     51.5102 |      58.0166 |      54.3933 |      54.3933 |       65.1207 |
| result/mask.target_word                                              |         35.4212 |    51.8862 |    51.8862 |     58.8232 |     54.4272 |     54.4272 |      65.2186 |            20.5 |    34.5833 |    34.5833 |     42.1285 |     38.9635 |     38.9635 |      53.0641 |         11.4754 |    22.541  |    22.541  |     28.1829 |     29.2006 |     29.2006 |      45.1744 |          22.4655 |     36.3369 |     36.3369 |      43.0449 |      40.8638 |      40.8638 |       54.4857 |
| result/This_is_mask..target_phrase                                   |         61.3391 |    73.7221 |    73.7221 |     78.6229 |     74.6562 |     74.6562 |      80.8288 |            23   |    39.675  |    39.675  |     47.8827 |     43.1677 |     43.1677 |      56.5028 |         23.6066 |    40.694  |    40.694  |     48.1727 |     44.585  |     44.585  |      57.6038 |          35.9819 |     51.3637 |     51.3637 |      58.2261 |      54.1363 |      54.1363 |       64.9784 |
| result/This_is_mask..target_word                                     |         34.3413 |    50.7883 |    50.7883 |     57.969  |     53.4535 |     53.4535 |      64.4897 |            19.5 |    33.7167 |    33.7167 |     41.0074 |     38.375  |     38.375  |      52.592  |         15.082  |    28.5137 |    28.5137 |     34.4631 |     34.7077 |     34.7077 |      49.6778 |          22.9744 |     37.6729 |     37.6729 |      44.4798 |      42.1787 |      42.1787 |       55.5865 |
