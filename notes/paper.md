# Paper 🩻 🐺 🤟🏽 😿


--------------------------------------------------
## Method

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
## Subpopulation Shift

| Subpopulation       | Unseen     | Seen       | Combined   |
| ------------------- | ---------: | ---------: | ---------: |
| Age [31-62]         | 87.50±0.23 | 93.76±0.12 | 91.83±0.13 |
| Age [10,30]∪[63,80] | 87.57±0.23 | 93.26±0.13 | 91.48±0.14 |
| Female              | 87.48±0.23 | 93.64±0.12 | 91.72±0.13 |
| Male                | 87.66±0.23 | 93.52±0.12 | 91.70±0.13 |
| AP                  | 89.20±0.22 | 93.49±0.14 | 92.04±0.14 |
| PA                  | 85.83±0.24 | 93.95±0.10 | 91.57±0.12 |
| Complete            |  |  |  |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Subdataset Shift

* Meta-trn is the same to complete
* Meta-val is the same when possible.
* Meta-tst consider clasees/examples only of the subdataset.

| Subdataset  | Unseen     | Seen       | Combined   |
| ----------- | ---------: | ---------: | ---------: |
| ChestXray14 | 91.89±0.18 | 93.52±0.11 | 92.76±0.12 |
| CheX        | 94.79±0.19 | 94.84±0.14 | 94.77±0.14 |
| MIMIC       | 93.81±0.19 | 94.95±0.13 | 94.47±0.13 |
| PadChest    | 81.71±0.66 | 95.49±0.10 | 95.37±0.09 |
| Complete    | 88.36±0.22 | 93.88±0.13 | 92.13±0.14 |


Observations
* La Beye es un 🐶


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


### 07/06 Friday


### 05/06 Thursday


### 04/06 Wednesday
* Results for SubPop/Complete wrote down
* Results for Arch wrote down
* Run EpisodeBased to check reproducibility

### 03/06 Tuesday
* Check generation of all resolutions for images 🤟🏽
* Commit last version of metachest repo 🤟🏽
* Run Complete Exp for SubPop 🤟🏽
* Results for SubDS/complete wrote down 🤟🏽
* Read ProtoNet 🤟🏽
* Setup ProtoNet analysis exp 🤟🏽

### 03/06 Monday
* Reproducibility verified for BatchBased 🤟🏽
* Run arch exp 🤟🏽
* Results for SubDS wrote down 🤟🏽
* Results for SubPop wrote down 🤟🏽
* Run Complete Exp for SubDS 🤟🏽
* Run generation of all resolutions for images 🤟🏽

### 31/05
* Read EpisodeBased 🤟🏽
* Check manual_backward on EpisodeBased 🤟🏽
* Setup EpisodeBased exp 🤟🏽

### 29/05
* Setup repo 🤟🏽
* Replicate repo on baymax 🤟🏽
* Run Subpopulation Shift exp 🤟🏽
* Finish results document 🤟🏽
