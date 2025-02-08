# Paper 🩻 🐺 🐶 🤟🏽 😿

## Experiments

Default HPs:

| HP                | Value     |
| :---------------- | --------: |
| Data              |           |
|   data_distro     | complete  |
|   mtrn_batch_size | 64        |
|   mtrn_n_way      | 3         |
|   mtrn_trn_k_shot | 5         |
|   mtrn_tst_k_shot | 15        |
|   mval_n_way      | 3         |
|   mval_n_unseen   | 1         |
|   mval_trn_k_shot | 5         |
|   mval_tst_k_shot | 15        |
|   mtst_n_way      | 3         |
|   mtst_n_unseen   | 1         |
|   mtst_trn_k_shot | 5         |
|   mtst_tst_k_shot | 15        |
|   image_size      | 384       |
|   data_aug        | false     |
| Network           |           |
|   net_backbone    | mobilenetv3-large-100 |
|   net_weights     | i1k       |
| Method            |           |
|   method          | batchbased |
|   batchbased_trn_lr | 0.0001  |
|   batchbased_sch_milestones | 1 |
|   batchbased_mval_lr | 0.005  |
|   batchbased_inner_steps | 100 |
|   batchbased_mval_net_steps | 0 |
|   batchbased_mval_net_lr | 0.005 |
|   batchbased_mval_net_batch_pct | 1.0 |
|   batchbased_mval_head_steps | 100 |
|   batchbased_mval_head_lr | 0.005 |
|   batchbased_mval_head_batch_pct | 0.5 |
|   batchbased_train_batches | 0 |
|   batchbased_reset_head | 0 |
| Training          |           |
|   mtrn_episodes   | 1000      |
|   mval_episodes   | 100       |
|   mtst_episodes   | 100000    |
|   max_epochs      | 150       |
|   stop_metric     | HM↑ |
|   stop_patience   | 10        |
|   deterministic   | warn      |
|   precision       | 16        |


--------------------------------------------------
### Method

- Data
    - complete distro has 146292 examples
    - batchbased needs around 150 epochs to converge
    - each epoch episodes only use 0.36 of the whole data wrt batches
    - each epoch episodes evaluates 2.76 faster wrt batches

|                | Size  | # per epoch | % used  | eval freq |
| -------------- | ----: | ----------: | ------: | --------: |
| Batch          | 64    | 2286        | 1       | 1         |
| Episode        | 52.96 | 1000        | 0.36    | 2.76      |


- Hyper-params
- MetaChest pretraining is a batch-based pretraning (BatchBased have it by design)


| Pretraining    |            |            |            |  MetaChest |  MetaChest | MetaChest  |
| -------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
|                | Unseen     | Seen       | HM   | Unseen     | Seen       | HM   |
| *STL-          |            |            |            |            |            |            |
|   BatchBased   |  |  |  |  |  |  |
|   EpisodeBased |  |  |  |  |  |  |
| *MTL-          |            |            |            |            |            |            |
|   ProtoNet     |  |  |  |  |  |  |
|   FEAT         |  |  |  |  |  |  |


Observations
- La Beye es un 🐶


--------------------------------------------------
### Pretraining

- Batchbased with/without MetaChest pretraning is the same as Batchbased is the method for pretraning.

| ImageNet | MetaChest  | Method       | Seen       | Unseen     | HM         |
| -------: | ---------: | -----------: | ---------: | ---------: | ---------: |
|          | ✅         | BatchBased   | 80.08±0.17 | 69.98±0.48 | 71.76±0.36 |
| 1k       | ✅         | BatchBased   | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| 21k      | ✅         | BatchBased   | 83.82±0.15 | 72.33±0.48 | 74.83±0.36 |
|          |            | ProtoNet     | 76.76±0.15 | 71.42±0.41 | 71.95±0.30 |
| 1k       |            | ProtoNet     | 82.05±0.15 | 76.20±0.38 | 77.44±0.27 |
| 21k      |            | ProtoNet     | 79.93±0.14 | 77.92±0.36 | 77.44±0.25 |
|          | ✅         | ProtoNet     | 79.24±0.14 | 76.18±0.38 | 76.10±0.26 |
| 1k       | ✅         | ProtoNet     | 81.70±0.14 | 79.71±0.36 | 79.29±0.25 |
| 21k      | ✅         | ProtoNet     | 81.58±0.14 | 77.04±0.36 | 77.79±0.26 |

Observations
- mtrn_episodes = number_of_examples / avg_size_episode = 4087
-


--------------------------------------------------
### Architecture

- TODO: Check ConvNext FLOPS
- Batch size 32

| Backbone              | Seen       | Unseen     | HM         | Params     | MACs (G)  | Encoding |
| --------------------- | ---------: | ---------: | ---------: | ---------: | --------: | -------: |
| Efficient             |            |            |            |            |           |          |
|   MobileNetV3Small075 | 83.39±0.16 | 72.74±0.47 | 74.90±0.36 |  1,016,584 |      0.11 |     1024 |
|   MobileViTv2-050     | 83.35±0.16 | 75.35±0.45 | 76.63±0.35 |  1,113,305 |      1.04 |      256 |
|   MobileNetV3Large100 | 84.42±0.15 | 74.70±0.47 | 76.57±0.36 |  4,201,744 |      0.62 |     1280 |
|   MobileViTv2-100     | 83.14±0.16 | 76.88±0.44 | 77.61±0.33 |  4,388,265 |      4.06 |      512 |
|   ConvNextAtto        | 83.54±0.15 | 73.00±0.44 | 75.44±0.34 |  3,373,240 |      1.62 |      320 |
| Large                 |            |            |            |            |           |          |
|   Densenet121         | 81.50±0.17 | 76.96±0.40 | 77.20±0.30 |  6,947,584 |      8.09 |     1024 |
|   Densenet161         | 83.36±0.16 | 77.83±0.42 | 78.41±0.31 | 26,462,592 |     22.36 |     2208 |
|   ConvNextTiny        | 85.26±0.14 | 76.11±0.44 | 78.07±0.34 | 27,817,056 |     18.36 |      768 |
|   MobileViTv2-200     | 82.98±0.16 | 77.58±0.43 | 77.96±0.33 | 17,423,177 |     16.07 |      512 |

Observations
- La Beye es un 🐶

| run                   | Seen       | Unseen     | HM         |
|:----------------------|:-----------|:-----------|:-----------|
| convnextv2-atto       | 83.90±0.15 | 77.04±0.44 | 77.94±0.34 |
| convnextv2-nano       | 82.72±0.16 | 74.59±0.44 | 76.09±0.33 |
| convnextv2-tiny       | 84.43±0.15 | 77.04±0.43 | 78.38±0.32 |


--------------------------------------------------
### RX Resolution

| Backbone            | Resolution | Seen       | Unseen     | HM         |
| ------------------: | ---------: | ---------: | ---------: | ---------: |
| MobileNetV3Large100 |        224 | 80.82±0.17 | 71.20±0.44 | 73.36±0.31 |
|                     |        384 | 84.42±0.15 | 75.25±0.46 | 76.95±0.35 |
|                     |        512 | 84.65±0.15 | 75.63±0.45 | 77.34±0.35 |
|                     |        768 | 83.69±0.15 | 74.99±0.46 | 76.55±0.35 |
|                     |       1024 | 84.45±0.15 | 75.34±0.46 | 76.99±0.35 |
| ConvNextTiny        |        224 | 84.58±0.15 | 74.76±0.44 | 76.95±0.34 |
|                     |        384 | 84.53±0.15 | 76.45±0.43 | 78.05±0.33 |
|                     |        512 | 84.75±0.15 | 76.58±0.43 | 78.17±0.33 |
|                     |        768 | 84.85±0.15 | 76.89±0.43 | 78.45±0.32 |
| Densenet121         |        224 | 83.03±0.16 | 76.84±0.42 | 77.62±0.32 |
|                     |        384 | 82.36±0.17 | 77.99±0.43 | 77.91±0.33 |
|                     |        512 | 82.33±0.17 | 77.60±0.41 | 77.92±0.30 |

Observations
- mtrn_batch_size=32 to fit in memory
- Resolution peeks (MH):
  - 512 - MobileNetV3Large100
  - 768 - ConvNextTiny
  - 512 - Densenet121
- ConvNextTiny outperforms (HM, 384):
  - MobileNetV3Large100 by 1.09
  - Densenet121 by 0.14


--------------------------------------------------
### Distribution Shift

#### Subdataset Shift

- Meta-trn is the same to complete
- Meta-val is the same when possible.
- Meta-tst consider clasees/examples only of the subdataset.

| Subdataset          | Seen       | Unseen     | HM         |
| ------------------- | ---------: | ---------: | ---------: |
| Complete            | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| ChestXray14         | 69.08±0.16 | 71.78±0.43 | 68.29±0.30 |
| CheXpert            | 74.87±0.20 | 80.67±0.25 | 77.08±0.20 |
| MIMIC               | 72.07±0.19 | 75.67±0.28 | 72.95±0.20 |
| PadChest            | 79.66±0.18 | 75.90±0.40 | 75.89±0.31 |

Observations
- Analyze how ech dataset result relats to the class distribution, imbalance, labeling quality

#### Subpopulation Shift

| Subpopulation       | Seen       | Unseen     | HM         |
| ------------------- | ---------: | ---------: | ---------: |
| Complete            | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| Age [31-62]         | 84.18±0.15 | 75.83±0.44 | 77.52±0.33 |
| Age [10,30]∪[63,80] | 84.56±0.14 | 73.83±0.45 | 76.28±0.35 |
| Female              | 83.70±0.15 | 75.52±0.45 | 76.97±0.34 |
| Male                | 84.83±0.14 | 74.17±0.45 | 76.58±0.35 |

Observations
- Age [31-62] > Age [10,30]∪[63,80] is expected
- Female > Male is unexpected

#### View Shift

| View                | Seen       | Unseen     | HM         |
| ------------------- | ---------: | ---------: | ---------: |
| Complete            | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| AP                  | 84.46±0.15 | 67.39±0.41 | 72.59±0.32 |
| PA                  | 82.61±0.16 | 77.31±0.45 | 77.43±0.34 |

Observations
- Age [31-62] > Age [10,30]∪[63,80] is expected
- Female > Male is unexpected


--------------------------------------------------
### From Generalized to Standard FSL

|   n-way |   n-unseen |   k-shot | Seen     | Unseen | HM    |
|--------:|-----------:|---------:|---------:|-------:|------:|
|       3 |          1 |        1 |    70.13 |  78.6  | 71.36 |
|       3 |          1 |        5 |    74.5  |  84.54 | 76.51 |
|       3 |          1 |       15 |    80.42 |  86.31 | 81.29 |
|       3 |          2 |        1 |    60.5  |  81.44 | 68.13 |
|       3 |          2 |        5 |    70.67 |  84.92 | 76.46 |
|       3 |          2 |       15 |    76.51 |  86.24 | 80.6  |
|       3 |          3 |        1 |    56.91 | nan    | 56.91 |
|       3 |          3 |        5 |    64.96 | nan    | 64.96 |
|       3 |          3 |       15 |    71.21 | nan    | 71.21 |
|       4 |          1 |        1 |    69.78 |  78.12 | 72.08 |
|       4 |          1 |        5 |    79.02 |  84.32 | 80.55 |
|       4 |          1 |       15 |    84.1  |  86.44 | 84.69 |
|       4 |          2 |        1 |    62.45 |  80.13 | 69.52 |
|       4 |          2 |        5 |    72.66 |  85.24 | 78.11 |
|       4 |          2 |       15 |    77.82 |  87.02 | 81.92 |
|       4 |          3 |        1 |    59.54 |  82.08 | 68.11 |
|       4 |          3 |        5 |    68.34 |  85.59 | 75.56 |
|       4 |          3 |       15 |    73.82 |  86.97 | 79.53 |
|       4 |          4 |        1 |    57.47 | nan    | 57.47 |
|       4 |          4 |        5 |    64.94 | nan    | 64.94 |
|       4 |          4 |       15 |    70.68 | nan    | 70.68 |
|       5 |          1 |        1 |    70.8  |  77.87 | 72.89 |
|       5 |          1 |        5 |    82.08 |  84.11 | 82.63 |
|       5 |          1 |       15 |    85.97 |  86.32 | 85.93 |
|       5 |          2 |        1 |    63.63 |  79.09 | 70.03 |
|       5 |          2 |        5 |    74.01 |  84.67 | 78.75 |
|       5 |          2 |       15 |    78.74 |  86.56 | 82.31 |
|       5 |          3 |        1 |    60.79 |  80.75 | 68.94 |
|       5 |          3 |        5 |    69.98 |  85.68 | 76.84 |
|       5 |          3 |       15 |    75.01 |  87.36 | 80.58 |
|       5 |          4 |        1 |    59.23 |  82.55 | 68.25 |
|       5 |          4 |        5 |    67.59 |  86.01 | 75.33 |
|       5 |          4 |       15 |    72.81 |  87.48 | 79.21 |
|       5 |          5 |        1 |    58.1  | nan    | 58.1  |
|       5 |          5 |        5 |    65.45 | nan    | 65.45 |
|       5 |          5 |       15 |    70.77 | nan    | 70.77 |


Observations
- La Beye es un 🐶


--------------------------------------------------
--------------------------------------------------
--------------------------------------------------

## TODO & Progress


### General Plan
- [ ] Update HPs
- [ ] Develop plots resolution, gfsl
- [ ] Check TIMM/MobileNet-V4, MamabaOut
        https://huggingface.co/collections/timm/mobilenetv4-pretrained-weights-6669c22cda4db4244def9637
- [ ] metachest-dev TODO.md
- [ ] https://pixi.sh/


### Progress

--------------------------------------------------
#### Feb 1st Week 09-15

--------------------------------------------------
#### Feb 1st Week 02-08
- [ ] RX Resolution Exp: analyze
- [ ] Pretraning Exp: prepare
- [ ] Pretraning Exp: run
- [ ] Pretraning Exp: analize



--------------------------------------------------
### Explore

### ImageNet vs Foundation

| Backbone              | Pretraining | Params     | MACs (G)  | Unseen     | Seen       | HM   |
| --------------------- | ----------: | ---------: | --------: | ---------: | ---------: | ---------: |
| MobileNetV3Large100   | I1K         |  4,201,744 |      0.62 |       1280 |  |  |  |
| MobileNetV3Large100   | I21K        |  4,201,744 |      0.62 |       1280 |  |  |  |
| EVA02-Tiny            | M38M        |  4,201,744 |      0.62 |       1280 |  |  |  |
| EVA02-Small           | M38M        |  4,201,744 |      0.62 |       1280 |  |  |  |

Observations
- Eva02 https://arxiv.org/pdf/2303.11331
- M38M (Merged-38M): I21K, CC12M, CC3M, COCO, ADE20K, Object365, OpenImages

- [ ] HP search EpisodeBased
- [ ] Check FOMAML https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial16/Meta_Learning.html
- [ ] Check Reptile or iMAML
- [ ] Check FEAT
- [ ] Check another method
- [ ] Review ProtoNet
- [ ] Review EpisodeBased
- [ ] Implemet EpisodeBased experiment with equivalence between batch and episode data size
