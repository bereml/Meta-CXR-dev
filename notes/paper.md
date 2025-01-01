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
|   stop_metric     | combined↑ |
|   stop_patience   | 10        |
|   deterministic   | warn      |
|   precision       | 16        |


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
|                | Unseen     | Seen       | Combined   | Unseen     | Seen       | Combined   |
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

| ImageNet | MetaChest  | Method       | Unseen     | Seen       | Combined   |
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

| Backbone              | Pretraining | Params     | MACs (G)  | Unseen     | Seen       | Combined   |
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

| Backbone            | Resolution | Unseen     | Seen       | Combined   |
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


| Backbone              | Unseen     | Seen       | Combined   | Params     | MACs (G)  | Encoding |
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

| Backbone              | Unseen     | Seen       | Combined   | Params     | MACs (G)  | Encoding |
| --------------------- | ---------: | ---------: | ---------: | ---------: | --------: | -------: |
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
## Subdataset Shift

* Meta-trn is the same to complete
* Meta-val is the same when possible.
* Meta-tst consider clasees/examples only of the subdataset.

| Subdataset          | Unseen     | Seen       | Combined   |
| ------------------- | ---------: | ---------: | ---------: |
| ChestXray14         |  |  |  |
| CheXpert            |  |  |  |
| MIMIC               |  |  |  |
| PadChest            |  |  |  |
| Complete            |  |  |  |


Observations
* La Beye es un 🐶


--------------------------------------------------
## Subpopulation Shift

| Subpopulation       | Unseen     | Seen       | Combined   |
| ------------------- | ---------: | ---------: | ---------: |
| Age [31-62]         |  |  |  |
| Age [10,30]∪[63,80] |  |  |  |
| Female              |  |  |  |
| Male                |  |  |  |
| Complete            |  |  |  |

Observations
* Anatomy Shift?


--------------------------------------------------
## View Shift

| Subpopulation       | Unseen     | Seen       | Combined   |
| ------------------- | ---------: | ---------: | ---------: |
| AP                  |  |  |  |
| PA                  |  |  |  |
| Complete            |  |  |  |


--------------------------------------------------
## From Generalized to Standard FSL

| n-way | n-unseen | Unseen     | Seen       | Combined   |
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



--------------------------------------------------
#### 01/06 - 01/12


--------------------------------------------------
#### 01/01 - 01/05
- [ ] Investigate div by zero at method/base.py:47
    python eval.py --results_dir rpaper --exp shift_ds --run ds_padchest --data_distro ds_padchest --seed 0 --checkpoint_name base
- [ ] Investigate why evaluation gives diferent results
    | run                | seen       | unseen     | hm         |
    | ------------------ | ---------- | ---------- | ---------- |
    | base               | 84.54±0.15 | 74.50±0.47 | 76.51±0.36 |
    | shift_ds/complete  | 79.74±0.18 | 69.16±0.44 | 71.67±0.31 |
    | shift_pop/complete | 79.74±0.18 | 69.16±0.44 | 71.67±0.31 |

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
