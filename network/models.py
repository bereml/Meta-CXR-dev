from os.path import isfile, join

import timm
from timm.models.helpers import load_checkpoint

from .backbones import BACKBONES


def _create_timm(_, model_name, weights, features_only):
    pretrained = not(weights == 'random')
    return timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=0,
        features_only=features_only
    )


def _create_with_checkpoint(_, model_name, weights, features_only):
    path = join('checkpoints', f'{weights}.pth')
    if not isfile(path):
        raise FileNotFoundError(f'Checkpoint not found {path}')
    model = timm.create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=0,
        features_only=features_only,
    )
    load_checkpoint(model, path, strict=False)
    return model


def _create(backbone, model_name, weights, features_only):
    if weights in {'random', 'i1k', 'i21k'}:
        create_fn = _create_timm
    else:
        create_fn = _create_with_checkpoint
    model = create_fn(backbone, model_name,
                      weights, features_only)
    return model


@BACKBONES.register('densenet121')
def densenet121(weights, features_only):
    model = _create('densenet121', 'densenet121',
                    weights, features_only)
    model.out_features = 1024
    return model


@BACKBONES.register('densenet161')
def densenet161(weights, features_only):
    model = _create('densenet161', 'densenet161',
                    weights, features_only)
    model.out_features = 2208
    return model


@BACKBONES.register('convnext-atto')
def convnext_atto(weights, features_only):
    model = _create('convnext-atto', 'convnext_atto',
                    weights, features_only)
    model.out_features = 320
    return model


@BACKBONES.register('convnext-tiny')
def convnext_tiny(weights, features_only):
    model = _create('convnext-tiny', 'convnext_tiny',
                    weights, features_only)
    model.out_features = 768
    return model


@BACKBONES.register('eva02-tiny')
def eva02_tiny(weights, features_only):
    model = _create('eva02-tiny',
                    'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
                    weights, features_only)
    model.out_features = 192
    return model


@BACKBONES.register('mobilenetv3-small-075')
def mobilenetv3_small_075(weights, features_only):
    model = _create('mobilenetv3-small-075', 'mobilenetv3_small_075',
                    weights, features_only)
    model.out_features = 1024
    return model


@BACKBONES.register('mobilenetv3-large-100')
def mobilenetv3_large_100(weights, features_only):
    model_name = {
        'i21k': 'mobilenetv3_large_100_miil_in21k',
    }.get(weights, 'mobilenetv3_large_100')
    model = _create('mobilenetv3-large', model_name,
                    weights, features_only)
    model.out_features = 1280
    return model


@BACKBONES.register('mobilevitv2-050')
def mobilevitv2_050(weights, features_only):
    model = _create('mobilevitv2-050', 'mobilevitv2_050',
                    weights, features_only)
    model.out_features = 256
    return model


@BACKBONES.register('mobilevitv2-100')
def mobilevitv2_100(weights, features_only):
    model = _create('mobilevitv2-100', 'mobilevitv2_100',
                    weights, features_only)
    model.out_features = 512
    return model


@BACKBONES.register('mobilevitv2-200')
def mobilevitv2_200(weights, features_only):
    model = _create('mobilevitv2-200', 'mobilevitv2_200',
                    weights, features_only)
    model.out_features = 1024
    return model


__all__ = [
    'densenet121', 'densenet161',
    'convnext_tiny', 'convnext_atto',
    'eva02_tiny',
    'mobilenetv3_small_075', 'mobilenetv3_large_100',
    'mobilevitv2_050', 'mobilevitv2_100', 'mobilevitv2_200',
]
