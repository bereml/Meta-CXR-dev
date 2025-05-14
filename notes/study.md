# General Study 🩻 🐺 🤟🏽 😿


--------------------------------------------------
## Arch

* Only features, no FC

| Backbone              | Params     | MACs (G)  | Encoding |
| --------------------- | ---------: | --------: | -------: |
| Efficient             |            |           |          |
|   MobileNetV3Small075 |  1,016,872 |      0.06 |     1024 |
|   MobileViTv2-050     |  1,113,593 |      1.05 |  |
|   MobileNetV3Large100 |  |  |  |
|   MobileViTv2-100     |  |  |  |
|   ConvNextAtto        |  |  |  |
| Large                 |            |           |          |
|   Densenet121         |  |  |  |
|   Densenet161         |  |  |  |
|   ConvNextTiny        |  |  |  |
|   MobileViTv2-200     |  |  |  |


| Backbone            | Params (M) | MACs (G)  | Encoding |
| ------------------- | ---------: | --------: | -------: |
| -- Efficient        |            |           |          |
| MobileNetV3Small075 |       1.02 |      0.12 |     1024 |
| MobileViTv2-050     |       1.11 |      1.05 |      256 |
| MobileNetV3Large100 |       4.20 |      0.63 |     1280 |
| ConvNextAtto        |       3.37 |      1.62 |      320 |
| ConvNextV2Atto      |       3.39 |      1.62 |      320 |
| MobileViTv2-100     |       4.39 |      4.08 |      512 |
| -- Large            |            |           |          |
| Densenet121         |       6.95 |      8.33 |     1024 |
| MobileViTv2-200     |      17.42 |     16.11 |     1024 |
| ConvNextV2Nano      |      14.98 |      7.21 |      640 |
| Densenet161         |      26.47 |     22.70 |     2208 |
| ConvNextTiny        |      27.82 |     28.60 |      768 |
| ConvNextV2Tiny      |      27.87 |     28.60 |      768 |


* torchinfo doesn't report correctly MACS for ConvNextV2Nano, ConvNextTiny, ConvNextV2Tiny. This numbers were taken from ConvNextV2 paper, we need to verify if FC macs need to be subtracted.


def summary(backbone, input_size=(1, 3, 384, 384), params_units='M', macs_units='G'):

    import network
    import torchinfo
    from torchinfo.enums import Units

    net = network.create_net(backbone)
    model_stats = torchinfo.summary(net, input_size=input_size)

    units = {
        'A': Units.AUTO,
        'M': Units.MEGABYTES,
        'G': Units.GIGABYTES,
        'T': Units.TERABYTES,
    }
    params_units = units.get(params_units, 'A')
    macs_units = units.get(macs_units, 'A')

    model_stats.formatting.params_units = params_units
    model_stats.formatting.macs_units = macs_units

    return model_stats



--------------------------------------------------
## Methods

### Protonet

| Type | Size | Seen       | Unseen     | HM         |
| ---: | ---: | ---------: | ---------: | ---------: |
| avg  |   96 | 80.16±0.14 | 79.02±0.36 | 78.17±0.25 |
| avg  |  128 | 81.92±0.14 | 79.14±0.36 | 79.13±0.26 |
| avg  |  144 | 80.61±0.15 | 77.90±0.37 | 77.70±0.27 |
| fc   |   96 | 80.81±0.14 | 79.06±0.36 | 78.51±0.25 |
| fc   |  128 | 82.05±0.15 | 76.20±0.38 | 77.44±0.27 |
| fc   |  144 | 81.12±0.14 | 77.46±0.37 | 77.76±0.26 |

Observations
* AVG is reproducible according to the experiment
* FC 96 or AVG 128

--------------------------------------------------
## Pretraining

ProtoNet with AVG 128

| run                        | seen       | unseen     | hm         |
|:---------------------------|:-----------|:-----------|:-----------|
| i1k+batchbased_protonet    | 81.74±0.14 | 76.66±0.37 | 77.58±0.27 |
| i1k+protonet_batchbased    | 83.08±0.16 | 73.25±0.44 | 75.48±0.32 |
| i1k_batchbased             | 82.32±0.16 | 71.77±0.46 | 74.02±0.34 |
| i1k_protonet               | 81.92±0.14 | 79.14±0.36 | 79.13±0.26 |
| i21k+batchbased_protonet   | 82.16±0.13 | 78.53±0.36 | 78.91±0.25 |
| i21k+protonet_batchbased   | 83.77±0.15 | 73.50±0.47 | 75.60±0.35 |
| i21k_batchbased            | 84.00±0.15 | 72.86±0.48 | 75.11±0.37 |
| i21k_protonet              | 79.52±0.14 | 77.95±0.37 | 77.22±0.26 |
| random+batchbased_protonet | 78.89±0.15 | 76.57±0.39 | 76.03±0.28 |
| random+protonet_batchbased | 78.43±0.18 | 67.69±0.47 | 69.71±0.36 |
| random_batchbased          | 79.78±0.18 | 69.68±0.48 | 71.41±0.37 |
| random_protonet            | 75.59±0.16 | 73.65±0.40 | 72.74±0.28 |
*



--------------------------------------------------
## Reproducibility



## Include/exclude train images

| run                  | seen       | unseen     | hm         |
|----------------------|------------|------------|------------|
| Include train images | 93.51±0.12 | 86.63±0.23 | 89.59±0.17 |
| Exclude train images | 79.86±0.32 | 75.37±0.35 | 76.13±0.30 |


## Using No-finding images

| run               | seen       | unseen     | hm         |
|-------------------|------------|------------|------------|
| random_batchbased | 78.01±0.22 | 77.64±0.28 | 77.23±0.23 |
| random_protonet   | 77.13±0.23 | 75.98±0.33 | 75.81±0.27 |

