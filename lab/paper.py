"""
Experiments for Publication
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, eval_model, train_model


SEEDS = [0]
RESULTS_DIR = 'rpaper'
DEBUG_PARAMS = {
    'batchbased_train_batches': 1,
    'mtrn_episodes': 1,
    'mval_episodes': 1,
    'mtst_episodes': 1,
    'max_epochs': 1,
}


def paper_base(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='base',
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'base'
    run = 'base'
    cfgs = seeds
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        seed = cfg
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_arch(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'arch'
    cfgs = list(product(
        # arch
        [
            'mobilenetv3-small-075',
            'mobilenetv3-large-100',
            'mobilevitv2-050',
            'mobilevitv2-100',
            'convnext-atto',
            'densenet121',
            'densenet161',
            'convnext-tiny',
            'mobilevitv2-200',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        backbone, seed = cfg
        run = '_'.join([
            backbone,
        ])
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=backbone,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_foundation(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        mtrn_batch_size=48,
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'foundation'
    cfgs = list(product(
        # image_size, net_backbone, net_weights
        [
            # [336, 'eva02-tiny', 'i21k'],
            # [336, 'eva02-small', 'i21k'],
            [448, 'eva02-large', 'mim_m38m_ft_in22k_in1k'],
            # [448, 'eva02-large', 'mim_in22k_ft_in22k_in1k'],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (image_size, net_backbone, net_weights), seed = cfg
        run = '_'.join([
            net_backbone,
            net_weights
        ])
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            image_size=image_size,
            net_backbone=net_backbone,
            net_weights=net_weights,
            mtrn_batch_size=mtrn_batch_size,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_nway_unseen(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='base',
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'nway-unseen'
    cfgs = list(product(
        # n_way, n_unseen
        [
            [3, 1],
            [3, 2],
            [3, 3],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (n_way, n_unseen), seed = cfg
        run = '_'.join([
            f'nway-{n_way}',
            f'unseen-{n_unseen}',
        ])
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mval_n_way=n_way,
            mval_n_unseen=n_unseen,
            mtst_n_way=n_way,
            mtst_n_unseen=n_unseen,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_resolution(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        mtrn_batch_size=32,
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'resolution'
    cfgs = list(product(
        # image_size, net_backbone
        [
            [ 224, 'mobilenetv3-large-100'],
            [ 384, 'mobilenetv3-large-100'],
            [ 512, 'mobilenetv3-large-100'],
            [ 768, 'mobilenetv3-large-100'],
            [1024, 'mobilenetv3-large-100'],
            [ 224, 'convnext-tiny'],
            [ 384, 'convnext-tiny'],
            [ 512, 'convnext-tiny'],
            [ 768, 'convnext-tiny'],
            [1024, 'convnext-tiny'],
            [ 224, 'densenet121'],
            [ 384, 'densenet121'],
            [ 512, 'densenet121'],
            [ 768, 'densenet121'],
            [1024, 'densenet121'],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (image_size, net_backbone), seed = cfg
        run = '_'.join([
            f'{image_size:04d}',
            net_backbone,
        ])
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            image_size=image_size,
            net_backbone=net_backbone,
            mtrn_batch_size=mtrn_batch_size,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_shift_ds(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='base',
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'shift_ds'
    cfgs = list(product(
        # data_distro
        [
            'ds_chestxray14',
            'ds_chexpert',
            'ds_mimic',
            'ds_padchest',
            'complete',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        data_distro, seed = cfg
        run = '_'.join([
            data_distro,
        ])
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_shift_pop(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='base',
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'shift_pop'
    cfgs = list(product(
        # data_distro
        [
            'age_center',
            'age_tails',
            'sex_female',
            'sex_male',
            'complete',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (data_distro, seed) = cfg
        run = '_'.join([
            data_distro,
        ])
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_shift_view(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='base',
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'shift_view'
    cfgs = list(product(
        # data_distro
        [
            'view_ap',
            'view_pa',
            'complete',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (data_distro, seed) = cfg
        run = '_'.join([
            data_distro,
        ])
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))
