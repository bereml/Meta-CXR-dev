# Paper 🩻 🐺 🐶 🤟🏽 😿

Default hyper-params:

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
## Subdataset Shift

* Meta-trn is the same to complete
* Meta-val is the same when possible.
* Meta-tst consider clasees/examples only of the subdataset.

| Subdataset          | Unseen     | Seen       | HM         |
| ------------------- | ---------: | ---------: | ---------: |
| Complete            | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| ChestXray14         | 69.08±0.16 | 71.78±0.43 | 68.29±0.30 |
| CheXpert            | 74.87±0.20 | 80.67±0.25 | 77.08±0.20 |
| MIMIC               | 72.07±0.19 | 75.67±0.28 | 72.95±0.20 |
| PadChest            | 79.66±0.18 | 75.90±0.40 | 75.89±0.31 |

Observations
* Analyze how ech dataset result relats to the class distribution, imbalance, labeling quality


--------------------------------------------------
## Subpopulation Shift

| Subpopulation       | Unseen     | Seen       | HM         |
| ------------------- | ---------: | ---------: | ---------: |
| Complete            | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| Age [31-62]         | 84.18±0.15 | 75.83±0.44 | 77.52±0.33 |
| Age [10,30]∪[63,80] | 84.56±0.14 | 73.83±0.45 | 76.28±0.35 |
| Female              | 83.70±0.15 | 75.52±0.45 | 76.97±0.34 |
| Male                | 84.83±0.14 | 74.17±0.45 | 76.58±0.35 |

Observations
* Age [31-62] > Age [10,30]∪[63,80] is expected
* Female > Male is unexpected


--------------------------------------------------
## View Shift

| View       | Unseen     | Seen       | HM         |
| ------------------- | ---------: | ---------: | ---------: |
| Complete            | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
| AP                  | 84.46±0.15 | 67.39±0.41 | 72.59±0.32 |
| PA                  | 82.61±0.16 | 77.31±0.45 | 77.43±0.34 |

Observations
* Age [31-62] > Age [10,30]∪[63,80] is expected
* Female > Male is unexpected

--------------------------------------------------
## From Generalized to Standard FSL

|   n-way |   n-unseen |   k-shot |   Unseen |   Seen |    HM |
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
|       5 |          3 |        1 |    60.79 |  80.75 | 68.94 |
|       5 |          3 |        5 |    69.98 |  85.68 | 76.84 |
|       5 |          3 |       15 |    75.01 |  87.36 | 80.58 |
|       5 |          4 |        1 |    59.23 |  82.55 | 68.25 |
|       5 |          4 |        5 |    67.59 |  86.01 | 75.33 |


| n-way | n-unseen | Unseen     | Seen       | HM         |
| ----: | -------: | ---------: | ---------: | ---------: |
| 3     | 1        |  |  |  |
|       | 2        |  |  |  |
|       | 3        |  |  |  |
| 4     | 1        |  |  |  |
|       | 2        |  |  |  |
|       | 3        |  |  |  |
|       | 4        |  |  |  |
| 5     | 1        |  |  |  |
|       | 2        |  |  |  |
|       | 3        |  |  |  |
|       | 4        |  |  |  |
|       | 5        |  |  |  |

Observations
* La Beye es un 🐶


--------------------------------------------------
--------------------------------------------------
--------------------------------------------------

--------------------------------------------------
## Method

* Data
    - complete distro has 146292 examples
    - batchbased needs around 150 epochs to converge
    - each epoch episodes only use 0.36 of the whole data wrt batches
    - each epoch episodes evaluates 2.76 faster wrt batches

|                | Size  | # per epoch | % used  | eval freq |
| -------------- | ----: | ----------: | ------: | --------: |
| Batch          | 64    | 2286        | 1       | 1         |
| Episode        | 52.96 | 1000        | 0.36    | 2.76      |


* Hyper-params




* MetaChest pretraining is a batch-based pretraning (BatchBased have it by design)


| Pretraining    |            |            |            |  MetaChest |  MetaChest | MetaChest  |
| -------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
|                | Unseen     | Seen       | HM   | Unseen     | Seen       | HM   |
| *STL*          |            |            |            |            |            |            |
|   BatchBased   |  |  |  |  |  |  |
|   EpisodeBased |  |  |  |  |  |  |
| *MTL*          |            |            |            |            |            |            |
|   ProtoNet     |  |  |  |  |  |  |
|   FEAT         |  |  |  |  |  |  |






Observations
* La Beye es un 🐶


--------------------------------------------------
## Pretraining

* Batchbased with/without MetaChest pretraning is the same as Batchbased is the method for pretraning.

| ImageNet | MetaChest  | Method       | Unseen     | Seen       | HM   |
| -------: | ---------: | -----------: | ---------: | ---------: | ---------: |
|          | ✅         | BatchBased   |  |  |  |
| 1k       | ✅         | BatchBased   |  |  |  |
| 21k      | ✅         | BatchBased   |  |  |  |
|          |            | ProtoNet     |  |  |  |
| 1k       |            | ProtoNet     |  |  |  |
| 21k      |            | ProtoNet     |  |  |  |
|          | ✅         | ProtoNet     |  |  |  |
| 1k       | ✅         | ProtoNet     |  |  |  |
| 21k      | ✅         | ProtoNet     |  |  |  |

Observations
* ProtoNet: 1k improves Random by 1.85
* ProtoNet: 1k+MetaChest improves 1k by 5.25
* ProtoNet: prototypes imrpove with MetaChest pretraning
* BatchBased: The improvement of 1k over Random is only 0.23


--------------------------------------------------
## ImageNet vs Foundation

| Backbone              | Pretraining | Params     | MACs (G)  | Unseen     | Seen       | HM   |
| --------------------- | ----------: | ---------: | --------: | ---------: | ---------: | ---------: |
| MobileNetV3Large100   | I1K         |  4,201,744 |      0.62 |       1280 |  |  |  |
| MobileNetV3Large100   | I21K        |  4,201,744 |      0.62 |       1280 |  |  |  |
| EVA02-Tiny            | M38M        |  4,201,744 |      0.62 |       1280 |  |  |  |
| EVA02-Small           | M38M        |  4,201,744 |      0.62 |       1280 |  |  |  |

Observations
* Eva02 https://arxiv.org/pdf/2303.11331
* M38M (Merged-38M): I21K, CC12M, CC3M, COCO, ADE20K, Object365, OpenImages



--------------------------------------------------
## RX Resolution

| Backbone            | Resolution | Unseen     | Seen       | HM   |
| ------------------: | ---------: | ---------: | ---------: | ---------: |
| MobileNetV3Large100 |        224 |  |  |  |
|                     |        384 |  |  |  |
|                     |        512 |  |  |  |
|                     |        768 |  |  |  |
|                     |       1024 |  |  |  |
| Densenet121         |        224 |  |  |  |
|                     |        384 |  |  |  |
|                     |        512 |  |  |  |
|                     |        768 |  |  |  |
|                     |       1024 |  |  |  |
| ConvNextTiny        |        224 |  |  |  |
|                     |        384 |  |  |  |
|                     |        512 |  |  |  |
|                     |        768 |  |  |  |
|                     |       1024 |  |  |  |


--------------------------------------------------
## Architecture

* TODO: Check ConvNext FLOPS

* Batch size 64, missing values were uneble to run due to memory


| Backbone              | Unseen     | Seen       | HM         | Params     | MACs (G)  | Encoding |
| --------------------- | ---------: | ---------: | ---------: | ---------: | --------: | -------: |
| Efficient             |            |            |            |            |           |          |
|   MobileNetV3Small075 | 84.42±0.25 | 92.76±0.14 | 90.24±0.15 |  1,016,584 |      0.11 |     1024 |
|   MobileViTv2-050     | 86.68±0.25 | 93.47±0.13 | 91.38±0.15 |  1,113,305 |      1.04 |      256 |
|   MobileNetV3Large100 | 87.99±0.22 | 94.11±0.12 | 92.17±0.13 |  4,201,744 |      0.62 |     1280 |
|   MobileViTv2-100     | 87.59±0.25 | 93.87±0.13 | 91.94±0.14 |  4,388,265 |      4.06 |      512 |
|   ConvNextAtto        | 88.24±0.23 | 94.96±0.11 | 92.99±0.12 |  3,373,240 |      1.62 |      320 |
| Large                 |            |            |            |            |           |          |
|   Densenet121         | 90.78±0.21 | 94.77±0.11 | 93.44±0.13 |  6,947,584 |      8.09 |     1024 |
|   Densenet161         |            |            |            | 26,462,592 |     22.36 |     2208 |
|   ConvNextTiny        | 89.62±0.23 | 94.99±0.11 | 93.46±0.12 | 27,817,056 |     18.36 |      768 |
|   MobileViTv2-200     |            |            |            | 17,423,177 |     16.07 |      512 |

* Batch size 48

| Backbone              | Unseen     | Seen       | HM         | Params     | MACs (G)  | Encoding |
| --------------------- | ---------: | ---------: | ---------: | ------Coyote01---: | --------: | -------: |
| Efficient             |            |            |            |            |           |          |
|   MobileNetV3Small075 |  |  |  |  1,016,584 |      0.11 |     1024 |
|   MobileViTv2-050     |  |  |  |  1,113,305 |      1.04 |      256 |
|   MobileNetV3Large100 |  |  |  |  4,201,744 |      0.62 |     1280 |
|   MobileViTv2-100     |  |  |  |  4,388,265 |      4.06 |      512 |
|   ConvNextAtto        |  |  |  |  3,373,240 |      1.62 |      320 |
| Large                 |            |            |            |            |           |          |
|   Densenet121         |  |  |  |  6,947,584 |      8.09 |     1024 |
|   Densenet161         |  |  |  | 26,462,592 |     22.36 |     2208 |
|   ConvNextTiny        |  |  |  | 27,817,056 |     18.36 |      768 |
|   MobileViTv2-200     |  |  |  | 17,423,177 |     16.07 |      512 |


Observations
* La Beye es un 🐶














--------------------------------------------------
--------------------------------------------------
--------------------------------------------------

## TODO

### Reunión planeación
- [ ] Bere: revisar literatura
- [ ] Bere: acordar plan general
- [ ] Bere: acordar sesiones de escritura

### Plan general
- [ ] Limpiar codigo, generar versión y correr 3 veces
- [ ] Preparar & correr mínimos: Subdataset Shift, Subpopulation Shift, GFSL -> SFSL
- [ ] Preparar & correr experimentos completos del artículo
- [ ] Revisar TIMM/MobileNet-V4, MamabaOut
        https://huggingface.co/collections/timm/mobilenetv4-pretrained-weights-6669c22cda4db4244def9637
- [ ] metachest-dev TODO.md
- [ ] https://pixi.sh/

### Experimentos
- [ ] Exp Methods SetUp
- [ ] Exp Methods Run
- [ ] Exp Methods Analize

- [ ] Exp Pretraining SetUp
- [ ] Exp Pretraining Run
- [ ] Exp Pretraining Analize

- [ ] Exp Architectures SetUp
- [ ] Exp Architectures Run
- [ ] Exp Architectures Analize

- [ ] Exp Image Resolution SetUp
- [ ] Exp Image Resolution Run
- [ ] Exp Image Resolution Analize

- [ ] Exp Subdataset Shift SetUp
- [ ] Exp Subdataset Shift Run
- [ ] Exp Subdataset Shift Analize

- [ ] Exp Subpopulation Shift SetUp
- [ ] Exp Subpopulation Shift Run
- [ ] Exp Subpopulation Shift Analize

- [ ] Exp View Shift SetUp
- [ ] Exp View Shift Run
- [ ] Exp View Shift Analize

- [ ] Exp GFSL -> SFSL SetUp
- [ ] Exp GFSL -> SFSL Run
- [ ] Exp GFSL -> SFSL Analize





seed = 0
| run            | seen       | unseen     | hm         |
|:---------------|:-----------|:-----------|:-----------|
| complete       | 79.74±0.18 | 69.16±0.44 | 71.67±0.31 |
| ds_chestxray14 | 62.14±0.17 | 64.58±0.37 | 61.70±0.24 |
| ds_chexpert    | 67.55±0.20 | 70.96±0.27 | 68.52±0.21 |
| ds_mimic       | 65.25±0.19 | 66.53±0.29 | 64.96±0.21 |
| ds_padchest    | 72.40±0.21 | 69.13±0.37 | 69.19±0.27 |

seed = 1
| run            | seen       | unseen     | hm         |
|:---------------|:-----------|:-----------|:-----------|
| complete       | 79.74±0.18 | 69.66±0.43 | 72.00±0.31 |
| ds_chestxray14 | 62.14±0.17 | 64.73±0.37 | 61.82±0.23 |
| ds_chexpert    | 67.50±0.20 | 71.07±0.27 | 68.56±0.20 |
| ds_mimic       | 65.29±0.19 | 66.76±0.29 | 65.06±0.21 |
| ds_padchest    | 72.40±0.21 | 68.64±0.38 | 68.85±0.28 |


seed = 0
| run               | seen       | unseen     | hm         |
|:------------------|:-----------|:-----------|:-----------|
| random_batchbased | 78.01±0.22 | 77.64±0.28 | 77.23±0.23 |
| random_protonet   | 77.13±0.23 | 75.98±0.33 | 75.81±0.27 |



--------------------------------------------------
#### 01/06 - 01/12


--------------------------------------------------
#### 01/01 - 01/05
--checkpoint_name base
- [ ] Investigate why evaluation gives diferent results than fulll traninig
  * The stop patience is bigger
| run        | hm         | max_epochs | stop_patience | best_epoch |
|:-----------|:-----------|:-----------|:--------------|:-----------|
| eval-only  | 72.00±0.31 |        500 |            25 |          2 |
| full-train | 77.23±0.23 |        150 |            50 |          5 |


- [ ] Save HP when eval



- [x] Investigate div by zero at method/base.py:47
    python eval.py --results_dir rpaper --exp shift_ds --run ds_padchest --data_distro ds_padchest --seed 0
- [x] Code refactor of metachest
- [x] Add results overview on md & tex formats


--------------------------------------------------
--------------------------------------------------
--------------------------------------------------
## Progress History

--------------------------------------------------
#### 25/10 Friday
- [x] Run an experiment dont excluding mtst in mtrn (modify _load_data)

--------------------------------------------------
#### 26/08 Friday
- [ ] Run exp: Resolution
- [ ] Run exp: Arch
- [ ] Run exp: Pretraining
- [ ] Run exp: Method
- [ ] Analize exp: DS Shift
- [ ] Analize exp: Pop Shift
- [ ] Run exp: View Shift
- [ ] Run exp: GFSL
- [ ] Check Pretraining vs Base
- [ ] HP search EpisodeBased
- [ ] Check FOMAML https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial16/Meta_Learning.html
- [ ] Check Reptile or iMAML
- [ ] Check FEAT
- [ ] Check another method
- [ ] Analize exp: Pretraining


#### 25/08 Thusday

#### 24/08 Wednesday

#### 23/08 Tuesday

#### 05/08 Monday
- [x] Review ProtoNet
- [ ] Review EpisodeBased
- [ ] Implemet EpisodeBased experiment with equivalence between batch and episode data size
- [ ] Ryun new episode generation method
- [ ] Analize more in depth only one class distro

--------------------------------------------------
#### 26/07 Friday

#### 25/07 Thusday

#### 24/07 Wednesday

#### 23/07 Tuesday
- [x] Run exp: DS Shift
- [x] Run exp: Pop Shift
- [x] Verify the experiments are ready
- [x] Check ProtoNet

#### 05/08 Monday
- [x] Implement episode generation with exclusion of mtrn only examples
- [x] Develop coocurrence plot
- [x] Clean cecav3: .config/projects , ssh keys

--------------------------------------------------
#### 05/07 Friday
- [ ] Analize viability of Foundation exp
- [ ] Check another method
- [ ] Implement methods exp
- [ ] Run methods exp
- [ ] Develop Resolution plot pipeline

#### 04/07 Thusday
- [ ] Run Resolution exp

#### 03/07 Wednesday
- [x] Run GFSL Exp
- [x] Analyze Pretraining Exp results

#### 02/07 Tuesday

#### 01/07 Monday

--------------------------------------------------
#### 28/06 Friday
- [x] Run Pretraining exp

#### 27/06 Thusday

#### 26/06 Wednesday
- [x] Update Arch Exp
- [x] Run Arch Exp

#### 25/06 Tuesday
- [x] Run Protonet exp
- [x] Implement Pretraining exp
- [x] Check EpisodeBased method
- [x] Run EpisodeBased study

#### 24/06 Monday
- [x] Check Protonet method

--------------------------------------------------
#### 21/06 Friday
- [x] Run arch exp

#### 20/06 Thusday
- [x] Run shift_pop exps
- [x] Analyze shift ds exps
- [x] Analyze nway exp

#### 19/06 Wednesday

#### 18/06 Tuesday
- [x] Improve shift ds/pop exps to only eval
- [x] Run shift_ds exps
- [x] Run nway exps
- [x] Analyze repro exp
- [x] Implement foundation exp

#### 17/06 Monday
- [x] Sync repos

--------------------------------------------------
#### 14/06 Friday
- [x] How many examples are per batch/episode?
- [x] How many batches are there in the train data loader?
- [x] Batchbased will use all the batches before validation?

#### 13/06 Thusday
- [x] Fix resizing of padchest images
- [x] Set default HP on args for data/method/precision
- [x] Document HP

#### 12/06 Wednesday

#### 11/06 Tuesday

#### 10/06 Monday

--------------------------------------------------
#### 07/06 Friday

#### 05/06 Thursday

#### 04/06 Wednesday
- [x] Results for SubPop/Complete wrote down
- [x] Unify batchbased precision code

#### 03/06 Tuesday
- [x] Check generation of all resolutions for images
- [x] Commit last version of metachest repo
- [x] Run Complete Exp for SubPop
- [x] Results for SubDS/complete wrote down
- [x] Read ProtoNet
- [x] Setup ProtoNet analysis exp

#### 03/06 Monday
- [x] Reproducibility verified for BatchBased
- [x] Run arch exp
- [x] Results for SubDS wrote down
- [x] Results for SubPop wrote down
- [x] Run Complete Exp for SubDS
- [x] Run generation of all resolutions for images
