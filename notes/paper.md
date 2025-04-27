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
|                | Seen       | Unseen     | HM         | Seen       | Unseen     | HM         |
| *STL-          |            |            |            |            |            |            |
|   BatchBased   |  |  |  |  |  |  |
|   EpisodeBased |  |  |  |  |  |  |
| *MTL-          |            |            |            |            |            |            |
|   ProtoNet     |  |  |  |  |  |  |
|   FEAT         |  |  |  |  |  |  |


| Method         | Seen       | Unseen     | HM         |
| -------------- | ---------: | ---------: | ---------: |
| *STL-          |            |            |            |
|   BatchBased   | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
|   EpisodeBased |  |  |  |
| *MTL-          |            |            |            |
|   ProtoNet     | 82.05±0.15 | 76.20±0.38 | 77.44±0.27 |
|   FEAT         |  |  |  |



Observations
- La Beye es un 🐶


--------------------------------------------------
### Pretraining

- Batchbased with/without MetaChest pretraning is the same as Batchbased is the method for pretraning.

| Weights | 1st Round  | 2nd Round  | Seen       | Unseen     | HH         |
|:--------|:-----------|:-----------|:-----------|:-----------|:-----------|
| Random  | Batchbased |            | 80.08±0.17 | 69.98±0.48 | 71.76±0.36 |
| Random  | Protonet   |            | 76.76±0.15 | 71.42±0.41 | 71.95±0.30 |
| I1k     | Batchbased |            | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| I1k     | Protonet   |            | 82.05±0.15 | 76.20±0.38 | 77.44±0.27 |
| I21k    | Batchbased |            | 83.82±0.15 | 72.33±0.48 | 74.83±0.36 |
| I21k    | Protonet   |            | 79.93±0.14 | 77.92±0.36 | 77.44±0.25 |
| Random  | Batchbased | Protonet   | 79.24±0.14 | 76.18±0.38 | 76.10±0.26 |
| Random  | Protonet   | Batchbased | 79.87±0.17 | 68.99±0.48 | 71.05±0.36 |
| I1k     | Batchbased | Protonet   | 81.70±0.14 | 79.71±0.36 | 79.29±0.25 |
| I1k     | Protonet   | Batchbased | 83.25±0.16 | 71.56±0.48 | 74.10±0.36 |
| I21k    | Batchbased | Protonet   | 81.58±0.14 | 77.04±0.36 | 77.79±0.26 |
| I21k    | Protonet   | Batchbased | 83.24±0.15 | 71.25±0.47 | 74.04±0.35 |

Observations
- mtrn_episodes = number_of_examples / avg_size_episode = 4087


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

| Subpopulation        | Seen       | Unseen     | HM         |
| -------------------- | ---------: | ---------: | ---------: |
| Complete             | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| Age [31-62]          | 84.18±0.15 | 75.83±0.44 | 77.52±0.33 |
| Age [10,30]∪[63,80]  | 84.56±0.14 | 73.83±0.45 | 76.28±0.35 |
| Age Young [10-20]    | 80.01±0.18 | 80.70±0.35 | 78.88±0.26 |
| Age Adult [21-60]    | 84.21±0.15 | 75.38±0.45 | 77.05±0.35 |
| Age Old   [61-80]    | 84.69±0.15 | 72.22±0.44 | 75.49±0.33 |
| Female               | 83.70±0.15 | 75.52±0.45 | 76.97±0.34 |
| Male                 | 84.83±0.14 | 74.17±0.45 | 76.58±0.35 |
| Age Decade 2 [10-20] | 80.01±0.18 | 80.70±0.35 | 78.88±0.26 |
| Age Decade 3 [21-30] | 83.17±0.16 | 81.28±0.40 | 80.26±0.30 |
| Age Decade 4 [31-40] | 83.68±0.15 | 80.95±0.38 | 80.60±0.28 |
| Age Decade 5 [41-50] | 83.50±0.15 | 80.53±0.41 | 80.04±0.31 |
| Age Decade 6 [51-60] | 83.21±0.15 | 74.31±0.41 | 76.46±0.30 |
| Age Decade 7 [61-70] | 84.19±0.15 | 72.63±0.44 | 75.61±0.33 |
| Age Decade 8 [71-80] | 84.27±0.16 | 72.54±0.43 | 75.53±0.33 |






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

|   n-way |   n-unseen |   k-shot | Unseen     | Seen       | HM         |
|--------:|-----------:|---------:|:-----------|:-----------|:-----------|
|       3 |          1 |        1 | 70.13±0.46 | 78.60±0.22 | 71.36±0.35 |
|       3 |          1 |        5 | 74.50±0.47 | 84.54±0.15 | 76.51±0.36 |
|       3 |          1 |       15 | 80.42±0.41 | 86.31±0.13 | 81.29±0.30 |
|       3 |          2 |        1 | 60.50±0.22 | 81.44±0.30 | 68.13±0.21 |
|       3 |          2 |        5 | 70.67±0.19 | 84.92±0.22 | 76.46±0.16 |
|       3 |          2 |       15 | 76.51±0.17 | 86.24±0.20 | 80.60±0.15 |
|       3 |          3 |        1 | 56.91±0.14 | nan        | 56.91±0.14 |
|       3 |          3 |        5 | 64.96±0.13 | nan        | 64.96±0.13 |
|       3 |          3 |       15 | 71.21±0.11 | nan        | 71.21±0.11 |
|       4 |          1 |        1 | 69.78±0.38 | 78.12±0.17 | 72.08±0.28 |
|       4 |          1 |        5 | 79.02±0.32 | 84.32±0.11 | 80.55±0.22 |
|       4 |          1 |       15 | 84.10±0.24 | 86.44±0.09 | 84.69±0.16 |
|       4 |          2 |        1 | 62.45±0.21 | 80.13±0.17 | 69.52±0.16 |
|       4 |          2 |        5 | 72.66±0.16 | 85.24±0.11 | 78.11±0.11 |
|       4 |          2 |       15 | 77.82±0.14 | 87.02±0.10 | 81.92±0.10 |
|       4 |          3 |        1 | 59.54±0.15 | 82.08±0.28 | 68.11±0.16 |
|       4 |          3 |        5 | 68.34±0.12 | 85.59±0.20 | 75.56±0.12 |
|       4 |          3 |       15 | 73.82±0.11 | 86.97±0.19 | 79.53±0.11 |
|       4 |          4 |        1 | 57.47±0.11 | nan        | 57.47±0.11 |
|       4 |          4 |        5 | 64.94±0.10 | nan        | 64.94±0.10 |
|       4 |          4 |       15 | 70.68±0.09 | nan        | 70.68±0.09 |
|       5 |          1 |        1 | 70.80±0.34 | 77.87±0.14 | 72.89±0.24 |
|       5 |          1 |        5 | 82.08±0.23 | 84.11±0.09 | 82.63±0.15 |
|       5 |          1 |       15 | 85.97±0.16 | 86.32±0.08 | 85.93±0.10 |
|       5 |          2 |        1 | 63.63±0.19 | 79.09±0.14 | 70.03±0.14 |P
|       5 |          2 |        5 | 74.01±0.14 | 84.67±0.09 | 78.75±0.09 |
|       5 |          2 |       15 | 78.74±0.12 | 86.56±0.07 | 82.31±0.08 |
|       5 |          3 |        1 | 60.79±0.15 | 80.75±0.16 | 68.94±0.12 |
|       5 |          3 |        5 | 69.98±0.12 | 85.68±0.10 | 76.84±0.08 |
|       5 |          3 |       15 | 75.01±0.10 | 87.36±0.09 | 80.58±0.07 |
|       5 |          4 |        1 | 59.23±0.12 | 82.55±0.26 | 68.25±0.14 |
|       5 |          4 |        5 | 67.59±0.10 | 86.01±0.19 | 75.33±0.10 |
|       5 |          4 |       15 | 72.81±0.08 | 87.48±0.17 | 79.21±0.09 |
|       5 |          5 |        1 | 58.10±0.10 | nan        | 58.10±0.10 |
|       5 |          5 |        5 | 65.45±0.08 | nan        | 65.45±0.08 |
|       5 |          5 |       15 | 70.77±0.07 | nan        | 70.77±0.07 |


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
