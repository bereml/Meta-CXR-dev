# Paper 🩻 🐺 🤟🏽





--------------------------------------------------
## Pretraining

| ImageNet | SCXR       | Meta-trn | Meta-tst     | Result     |
| -------: | ---------: | -------: | -----------: | ---------: |
|          |            |          |              | P          |
| 1k       |            |          | EpisodeBased | 😿         |
| 2k       |            |          | EpisodeBased | 😿         |
|          | BatchBased |          | BatchBased   |            |
| 1k       | BatchBased |          | BatchBased   |            |
| 21k      | BatchBased |          | BatchBased   |            |
|          |            |          |              | MT         |
|          |            | ProtoNet | ProtoNet     |            |
|          |            |          |              | P+MT       |
| 1k       |            | ProtoNet | ProtoNet     |            |
| 21k      |            | ProtoNet | ProtoNet     |            |
|          | BatchBased | ProtoNet | ProtoNet     |            |
| 1k       | BatchBased | ProtoNet | ProtoNet     |            |
| 21k      | BatchBased | ProtoNet | ProtoNet     |            |



--------------------------------------------------
## Resolution

| Resolution | MobileNetV3Small | MobileNetV3Large | DenseNet121 |
| ---------: | ---------: | ---------: | ---------: |
| 224        |  |  |  |
| 384        |  |  |  |
| 512        |  |  |  |
| 768        |  |  | 😿         |
| 1024       |  | 😿         | 😿         |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Architecture

| Backbone            | BatchBased | Params     | MACs (G)  | Encoding |
| ------------------- | ---------: | ---------: | --------: | -------: |
| MobileNetV3Small075 |            |  1,016,584 |      0.11 |     1024 |
| MobileNetV3Large100 |            |  4,201,744 |      0.62 |     1280 |
| MobileViTv2-050     |            |  1,113,305 |      1.04 |      256 |
| MobileViTv2-100     |            |  4,388,265 |      4.06 |      512 |
| ConvNextAtto        |            |  3,373,240 |      1.61 |      320 |
| Large               |            |            |           |          |
| Densenet121         |            |  6,947,584 |      8.09 |     1024 |
| Densenet161         |            | 26,462,592 |     22.36 |     2208 |
| ConvNextTiny        |            | 27,817,056 |      0.87 |      768 |
| MobileViTv2-200     |            | 17,423,177 |     16.07 |      512 |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Subpopulation Shift

| Subpop   |      | meta-trn        | meta-tst        | uAP        |
| -------- | ---- | --------------- | --------------- | ---------: |
| Sex      | Same | Men             | Men             |            |
|          |      | Women           | Women           |            |
|          | Diff | Men             | Women           |            |
|          |      | Women           | Men             |            |
| View     | Same | AP              | AP              |            |
|          |      | PA              | PA              |            |
|          | Diff | AP              | PA              |            |
|          |      | PA              | AP              |            |
| Age      | Same | [10,30]∪[63,80] | [10,30]∪[63,80] |            |
|          |      | [31-62]         | [31,62]         |            |
|          | Diff | [10,30]∪[63,80] | [31,62]         |            |
|          |      | [31,62]         | [10,30]∪[63,80] |            |
| Complete | Same | All             | All             |            |


Observations
* La Beye es un 🐶


--------------------------------------------------
## Subdataset Shift

| meta-trn | meta-tst | uAP        |
| -------- | ---------| ---------: |
| SCXR     | CheX     |            |
|          | MIMIC    |            |
|          | NIH      |            |
|          | PadChest |            |
|          | SCXR     |            |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Method

| Method            |  5-shot    | 10-shot    | 20-shot    | 40-shot    | 60-shot    |
| ----------------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| BatchBased        |            |            |            |            |            |
| EpisodeBased      |            |            |            |            |            |
| ProtoNet          |            |            |            |            |            |
| FEAT              |            |            |            |            |            |
| ProtoBatchBased   |            |            |            |            |            |
| ProtoEpisodeBased |            |            |            |            |            |
| MAML              |            |            |            |            |            |
| Pachini           |            |            |            |            |            |

Observations
* La Beye es un 🐶


--------------------------------------------------
## From Generalized to Standard FSL

At least 5-shot examples.

| Unseen |  1         | 2          | 3          | 4          | 5          |
| -----: | ---------: | ---------: | ---------: | ---------: | ---------: |
| 3-way  |            |            |            |            |            |
| 4-way  |            |            |            |            |            |
| 5-way  |            |            |            |            |            |


--------------------------------------------------
## ImageNet vs Foundation


| Unseen |  1         | 2          | 3          | 4          | 5          |
| -----: | ---------: | ---------: | ---------: | ---------: | ---------: |
| 3-way  |            |            |            |            |            |
| 4-way  |            |            |            |            |            |
| 5-way  |            |            |            |            |            |





























## Record

### Today
* Implement FEAT
* Run ProtoNet HP Exp (Small emb and FC)
* Run Arch Exp 🐶
* Run EpisodeBased HP Exp 🐶
* Study Bachbased MultiStep 🐶
* Implement 16bit Precision 🐶
* Run Subds Table 🐶
* Run Pretraning Table 🐶
* Prevent rlab to be deleted

* Explore HPs for EpisodeBased
* Run another Pretrain Exp 🐶
* Implement resampling 🐶 😿
* Implement part of Pretrain table
* Run Archs on baymax  with batch 32
* Remove 1 channel code


| meta     | episode | RelNet  |
| -------: | ------: | ------: |
| meta-trn | trn     | sample  |
|          | tst     | query   |
| meta-val | trn     | support |
|          | tst     | test    |
| meta-tst | trn     | support |
|          | tst     | test    |


### TODO 🐶
* Run Pretraining table
* Implement and explore hyper-params for
  - EpisodeBased
  - ProtoNet 🐶
  - MAML
  - FEAT
  - Pachini
* Investigate efficient DenseNet
  - https://arxiv.org/pdf/1707.06990.pdf
  - https://github.com/gpleiss/efficient_densenet_pytorch
* Investigate why STD converges to .23
* Investigate why Convnext does so badly
* Check EVA
  - https://arxiv.org/pdf/2211.07636.pdf
  - https://github.com/baaivision/EVA

### Done 🐺
* Download and preprocess datasets
* Generate SuperChestXRay and partitions


--------------------------------------------------
## Pretraining

| ImageNet | SCXR       | Meta-trn | Meta-tst     | Result     |
| -------: | ---------: | -------: | -----------: | ---------: |
|          |            |          |              | P          |
| 1k       |            |          | EpisodeBased | 😿         |
| 2k       |            |          | EpisodeBased | 😿         |
|          | BatchBased |          | BatchBased   |            |
| 1k       | BatchBased |          | BatchBased   |            |
| 21k      | BatchBased |          | BatchBased   |            |
|          |            |          |              | MT         |
|          |            | ProtoNet | ProtoNet     |            |
|          |            |          |              | P+MT       |
| 1k       |            | ProtoNet | ProtoNet     |            |
| 21k      |            | ProtoNet | ProtoNet     |            |
|          | BatchBased | ProtoNet | ProtoNet     |            |
| 1k       | BatchBased | ProtoNet | ProtoNet     |            |
| 21k      | BatchBased | ProtoNet | ProtoNet     |            |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Resolution

| Resolution | MobileNetV3Small | MobileNetV3Large | DenseNet121 |
| ---------: | ---------: | ---------: | ---------: |
| 224        |  |  |  |
| 384        |  |  |  |
| 512        |  |  |  |
| 768        |  |  | 😿         |
| 1024       |  | 😿         | 😿         |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Architecture

| Backbone            | BatchBased | Params     | MACs (G)  | Encoding |
| ------------------- | ---------: | ---------: | --------: | -------: |
| MobileNetV3Small075 |            |  1,016,584 |      0.11 |     1024 |
| MobileNetV3Large100 |            |  4,201,744 |      0.62 |     1280 |
| MobileViTv2-050     |            |  1,113,305 |      1.04 |      256 |
| MobileViTv2-100     |            |  4,388,265 |      4.06 |      512 |
| ConvNextAtto        |            |  3,373,240 |      1.61 |      320 |
| Large               |            |            |           |          |
| Densenet121         |            |  6,947,584 |      8.09 |     1024 |
| Densenet161         |            | 26,462,592 |     22.36 |     2208 |
| ConvNextTiny        |            | 27,817,056 |      0.87 |      768 |
| MobileViTv2-200     |            | 17,423,177 |     16.07 |      512 |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Subpopulation Shift

| Subpop   |      | meta-trn        | meta-tst        | uAP        |
| -------- | ---- | --------------- | --------------- | ---------: |
| Sex      | Same | Men             | Men             |            |
|          |      | Women           | Women           |            |
|          | Diff | Men             | Women           |            |
|          |      | Women           | Men             |            |
| View     | Same | AP              | AP              |            |
|          |      | PA              | PA              |            |
|          | Diff | AP              | PA              |            |
|          |      | PA              | AP              |            |
| Age      | Same | [10,30]∪[63,80] | [10,30]∪[63,80] |            |
|          |      | [31-62]         | [31,62]         |            |
|          | Diff | [10,30]∪[63,80] | [31,62]         |            |
|          |      | [31,62]         | [10,30]∪[63,80] |            |
| Complete | Same | All             | All             |            |


Observations
* La Beye es un 🐶


--------------------------------------------------
## Subdataset Shift

| meta-trn | meta-tst | uAP        |
| -------- | ---------| ---------: |
| SCXR     | CheX     |            |
|          | MIMIC    |            |
|          | NIH      |            |
|          | PadChest |            |
|          | SCXR     |            |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Method

| Method            |  5-shot    | 10-shot    | 20-shot    | 40-shot    | 60-shot    |
| ----------------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| BatchBased        |            |            |            |            |            |
| EpisodeBased      |            |            |            |            |            |
| ProtoNet          |            |            |            |            |            |
| FEAT              |            |            |            |            |            |
| ProtoBatchBased   |            |            |            |            |            |
| ProtoEpisodeBased |            |            |            |            |            |
| MAML              |            |            |            |            |            |
| Pachini           |            |            |            |            |            |

Observations
* La Beye es un 🐶


--------------------------------------------------
## From Generalized to Standard FSL

At least 5-shot examples.

| unseen |  1         | 2          | 3          | 4          | 5          |
| -----: | ---------: | ---------: | ---------: | ---------: | ---------: |
| 3-way  |            |            |            |            |            |
| 5-way  |            |            |            |            |            |

Observations
* La Beye es un 🐶
* Number of evaluation with a seed:
```bash
In [7]: edf[edf.seed == 0].count()
Out[7]:
seed                  10000
uap                   10000
cardiomegaly           2060
edema                  2072
pneumothorax           1968
consolidation          1929
pneumonia              1971
effusion               2821
lung_opacity           2829
atelectasis            2871
infiltration           2895
nodule                 2844
mass                   2853
pleural_thickening     2887
```



## Record

## 03/04
* Pretraining: Training Cycle 🐶
* Pretraining: Scripts to Save Pretrained model 🐶
* Pretraining: Check Dataset code 🐶
* Pretraining: Check MobileNetV3Large loading 🐶
* Agree paper objectives 🐶