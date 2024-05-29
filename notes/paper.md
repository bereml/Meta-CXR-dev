# Paper 🩻 🐺 🤟🏽 😿

## Today
* Setup repo
* Replicate repo on baymax
* Finish results document
* Run Subpopulation Shift exp


## TODO
* Run experiments
* Check another arch
* Verify reproducibility
* Develop nb for plot results
* Explore another method
* Develop Foundation exp
* Implement snakemake pipeline
* Write README


--------------------------------------------------
## Method

| Backbone     | Unseen     | Seen       | Combined   | Unseen     | Seen       | Combined   |
| ------------ | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
|              |       None |       None |       None |  MetaChest |  MetaChest | MetaChest  |
| BatchBased   | 89.52±0.21 | 93.40±0.13 | 92.00±0.14 | 89.52±0.21 | 93.40±0.13 | 92.00±0.14 |
| ProtoNet     | 74.35±0.33 | 86.30±0.24 | 82.69±0.24 | 78.11±0.33 | 90.67±0.20 | 87.13±0.22 |
| FEAT         | 72.19±0.36 | 79.73±0.24 | 77.45±0.23 | 72.34±0.40 | 82.45±0.25 | 79.46±0.25 |
| EpisodeBased | 60.65±0.28 | 62.96±0.20 | 62.44±0.18 | 74.21±0.29 | 81.22±0.19 | 78.96±0.18 |


Observations
* La Beye es un 🐶


--------------------------------------------------
## Pretraining

* Batchbased with/without MetaChest pretraning is the same as Batchbased is the method for pretraning.

| ImageNet | MetaChest  | Method       | Unseen     | Seen       | Combined   |
| -------: | ---------: | -----------: | ---------: | ---------: | ---------: |
|          | ✅         | BatchBased   | 89.11±0.21 | 93.38±0.13 | 91.87±0.14 |
| 1k       | ✅         | BatchBased   | 88.61±0.22 | 94.27±0.12 | 92.49±0.13 |
| 21k      | ✅         | BatchBased   | 87.09±0.23 | 93.77±0.13 | 91.65±0.14 |
|          |            | ProtoNet     | 75.92±0.32 | 83.77±0.25 | 81.27±0.25 |
| 1k       |            | ProtoNet     | 74.88±0.33 | 88.35±0.23 | 84.35±0.23 |
| 21k      |            | ProtoNet     | 75.18±0.34 | 89.31±0.23 | 85.15±0.23 |
|          | ✅         | ProtoNet     | 81.47±0.30 | 93.97±0.15 | 90.86±0.16 |
| 1k       | ✅         | ProtoNet     | 79.47±0.30 | 95.06±0.12 | 91.47±0.14 |
| 21k      | ✅         | ProtoNet     | 78.53±0.31 | 94.23±0.14 | 90.44±0.17 |

Observations
* La Beye es un 🐶


--------------------------------------------------
## ImageNet vs Foundation

* Batchbased with/without MetaChest pretraning is the same as Batchbased is the method for pretraning.

| ImageNet | MetaChest  | Method       | Unseen     | Seen       | Combined   |
| -------: | ---------: | -----------: | ---------: | ---------: | ---------: |
|          | ✅         | BatchBased   | 89.11±0.21 | 93.38±0.13 | 91.87±0.14 |
| 1k       | ✅         | BatchBased   | 88.61±0.22 | 94.27±0.12 | 92.49±0.13 |
| 21k      | ✅         | BatchBased   | 87.09±0.23 | 93.77±0.13 | 91.65±0.14 |
|          |            | ProtoNet     | 75.92±0.32 | 83.77±0.25 | 81.27±0.25 |
| 1k       |            | ProtoNet     | 74.88±0.33 | 88.35±0.23 | 84.35±0.23 |
| 21k      |            | ProtoNet     | 75.18±0.34 | 89.31±0.23 | 85.15±0.23 |
|          | ✅         | ProtoNet     | 81.47±0.30 | 93.97±0.15 | 90.86±0.16 |
| 1k       | ✅         | ProtoNet     | 79.47±0.30 | 95.06±0.12 | 91.47±0.14 |
| 21k      | ✅         | ProtoNet     | 78.53±0.31 | 94.23±0.14 | 90.44±0.17 |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Architecture

# TODO: Check ConvNext FLOPS

| Backbone            | Params     | MACs (G)  | Encoding | Unseen     | Seen       | Combined   |
| ------------------- | ---------: | --------: | -------: | ---------: | ---------: | ---------: |
|   Efficient         |            |           |          |            |            |            |
| MobileNetV3Small075 |  1,016,584 |      0.11 |     1024 | 77.41±0.88 | 82.70±0.61 | 81.77±0.55 |
| MobileViTv2-050     |  1,113,305 |      1.04 |      256 | 84.80±0.80 | 92.38±0.43 | 90.08±0.47 |
| MobileNetV3Large100 |  4,201,744 |      0.62 |     1280 | 87.66±0.75 | 94.36±0.37 | 92.29±0.43 |
| MobileViTv2-100     |  4,388,265 |      4.06 |      512 | 87.07±0.79 | 93.64±0.40 | 91.58±0.46 |
| ConvNextAtto        |  3,373,240 |      1.62 |      320 | 87.66±0.71 | 94.83±0.37 | 92.73±0.41 |
|   Large             |            |           |          |            |            |            |
| Densenet121         |  6,947,584 |      8.09 |     1024 | 89.51±0.67 | 94.70±0.36 | 93.05±0.40 |
| Densenet161         | 26,462,592 |     22.36 |     2208 | 89.68±0.70 | 94.32±0.38 | 92.83±0.42 |
| ConvNextTiny        | 27,817,056 |     18.36 |      768 | 88.90±0.72 | 94.89±0.35 | 93.15±0.40 |
| MobileViTv2-200     | 17,423,177 |     16.07 |      512 | 87.53±0.81 | 94.22±0.37 | 92.24±0.43 |

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
| AP                  |  |  |  |
| PA                  |  |  |  |
| Complete            |  |  |  |

Observations
* La Beye es un 🐶


--------------------------------------------------
## Subdataset Shift

* Meta-trn is the same to complete
* Meta-val is the same when possible.
* Meta-tst consider clasees/examples only of the subdataset.

| Subdataset | Seen       | Unseen     | Combined   |
| ---------- | ---------: | ---------: | ---------: |
| CheX       | 94.49±0.20 | 95.10±0.15 | 94.84±0.15 |
| MIMIC      | 93.28±0.20 | 95.19±0.13 | 94.48±0.14 |
| NIH        | 91.29±0.20 | 94.05±0.11 | 92.98±0.12 |
| PadChest   | 79.77±0.65 | 96.03±0.09 | 95.20±0.10 |
| MetaChest  | 86.74±0.24 | 93.84±0.13 | 91.68±0.14 |


Observations
* La Beye es un 🐶


--------------------------------------------------
## n-way & n-unseen

| n-way | n-unseen | Seen       | Unseen     | Combined   |
| ----: | -------: | ---------: | ---------: | ---------: |
| 3     | 1        | 86.74±0.24 | 93.84±0.13 | 91.68±0.14 |
|       | 2        | 69.83±0.21 | 86.77±0.29 | 76.88±0.22 |
|       | 3        |            | 63.30±0.15 | 63.30±0.15 |
| 4     | 1        | 83.55±0.22 | 95.18±0.07 | 92.84±0.09 |
|       | 2        | 72.49±0.18 | 91.27±0.16 | 83.37±0.16 |
|       | 3        | 66.33±0.15 | 83.88±0.29 | 72.20±0.17 |
|       | 4        |            | 61.82±0.11 | 61.82±0.11 |
| 5     | 1        | 80.51±0.22 | 95.51±0.05 | 93.26±0.06 |
|       | 2        | 73.45±0.16 | 92.68±0.10 | 86.47±0.11 |
|       | 3        | 67.50±0.14 | 89.72±0.18 | 78.34±0.15 |
|       | 4        | 62.22±0.12 | 85.15±0.31 | 68.78±0.17 |
|       | 5        |            | 57.39±0.09 | 57.39±0.09 |

Observations
* La Beye es un 🐶
