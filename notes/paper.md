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
* La Beye es un 🐶


--------------------------------------------------
## ImageNet vs Foundation

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
* La Beye es un 🐶


--------------------------------------------------
## Architecture

# TODO: Check ConvNext FLOPS

| Backbone              | Params     | MACs (G)  | Encoding | Unseen     | Seen       | Combined   |
| --------------------- | ---------: | --------: | -------: | ---------: | ---------: | ---------: |
| Efficient             |            |           |          |            |            |            |
|   MobileNetV3Small075 |  1,016,584 |      0.11 |     1024 |  |  |  |
|   MobileViTv2-050     |  1,113,305 |      1.04 |      256 |  |  |  |
|   MobileNetV3Large100 |  4,201,744 |      0.62 |     1280 |  |  |  |
|   MobileViTv2-100     |  4,388,265 |      4.06 |      512 |  |  |  |
|   ConvNextAtto        |  3,373,240 |      1.62 |      320 |  |  |  |
| Large                 |            |           |          |            |            |            |
|   Densenet121         |  6,947,584 |      8.09 |     1024 |  |  |  |
|   Densenet161         | 26,462,592 |     22.36 |     2208 |  |  |  |
|   ConvNextTiny        | 27,817,056 |     18.36 |      768 |  |  |  |
|   MobileViTv2-200     | 17,423,177 |     16.07 |      512 |  |  |  |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Subdataset Shift

* Meta-trn is the same to complete
* Meta-val is the same when possible.
* Meta-tst consider clasees/examples only of the subdataset.

| Subdataset          | Unseen     | Seen       | Combined   |
| ------------------- | ---------: | ---------: | ---------: |
| ChestXray14         | 91.89±0.18 | 93.52±0.11 | 92.76±0.12 |
| CheX                | 94.79±0.19 | 94.84±0.14 | 94.77±0.14 |
| MIMIC               | 93.81±0.19 | 94.95±0.13 | 94.47±0.13 |
| PadChest            | 81.71±0.66 | 95.49±0.10 | 95.37±0.09 |
| Complete            | 88.36±0.22 | 93.88±0.13 | 92.13±0.14 |


Observations
* Different results between Subdataset & Subpopulation


--------------------------------------------------
## Subpopulation Shift

| Subpopulation       | Unseen     | Seen       | Combined   |
| ------------------- | ---------: | ---------: | ---------: |
| Age [31-62]         | 87.50±0.23 | 93.76±0.12 | 91.83±0.13 |
| Age [10,30]∪[63,80] | 87.57±0.23 | 93.26±0.13 | 91.48±0.14 |
| Female              | 87.48±0.23 | 93.64±0.12 | 91.72±0.13 |
| Male                | 87.66±0.23 | 93.52±0.12 | 91.70±0.13 |
| AP                  | 89.20±0.22 | 93.49±0.14 | 92.04±0.14 |
| PA                  | 85.83±0.24 | 93.95±0.10 | 91.57±0.12 |
| Complete            | 87.95±0.22 | 93.71±0.13 | 91.84±0.13 |

Observations
* Different results between Subdataset & Subpopulation


--------------------------------------------------
## n-way & n-unseen

| n-way | n-unseen | Unseen     | Seen       | Combined   |
| ----: | -------: | ---------: | ---------: | ---------: |
| 3     | 1        |  |  |  |
|       | 2        |  |  |  |
|       | 3        |            |  |  |
| 4     | 1        |  |  |  |
|       | 2        |  |  |  |
|       | 3        |  |  |  |
|       | 4        |            |  |  |
| 5     | 1        |  |  |  |
|       | 2        |  |  |  |
|       | 3        |  |  |  |
|       | 4        |  |  |  |
|       | 5        |            |  |  |

Observations
* La Beye es un 🐶




--------------------------------------------------
--------------------------------------------------
--------------------------------------------------

## TODO
* Run experiments
* Check another arch
* Verify reproducibility
* Develop nb for plot results
* Explore another method
* Develop Foundation exp
* Implement snakemake pipeline
* Write README


#### 21/06 Friday
- [ ] Something

#### 20/06 Thusday

#### 19/06 Wednesday

#### 18/06 Tuesday

#### 17/06 Monday
- [ ] Check repro SubDS/SubPop Exp
- [ ] Results for Arch wrote down

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

#### 31/05
- [x] Read EpisodeBased
- [x] Check manual_backward on EpisodeBased
- [x] Setup EpisodeBased exp

#### 29/05
- [x] Setup repo
- [x] Replicate repo on baymax
- [x] Run Subpopulation Shift exp
- [x] Finish results document
