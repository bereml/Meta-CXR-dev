"""
Experiments for Publication
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, train_model


SEEDS = [0]
RESULTS_DIR = 'rpaper'



def paper_arch(
        mval_episodes=100,
        mtst_episodes=10000,
        weights='i1k',
        method='batchbased',
        # batchbased_mval_lr=0.005,
        # batchbased_inner_steps=100,
        max_epochs=150,
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    batchbased_train_batches = 0
    if debug:
        batchbased_train_batches = 2
        mval_episodes = 2
        mtst_episodes = 2
        max_epochs = 2
        results_dir = 'rdev'
    exp = f'arch'
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
            method,
            backbone,
        ])
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=backbone,
            net_weights=weights,
            method=method,
            batchbased_batch_size=48,
            # batchbased_mval_lr=batchbased_mval_lr,
            # batchbased_inner_steps=batchbased_inner_steps,
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_resolution(
        mval_episodes=100,
        mtst_episodes=1000,
        net_weights='random',
        method='batchbased',
        max_epochs=50,
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    batchbased_train_batches = 0
    if debug:
        batchbased_train_batches = 2
        mval_episodes = 2
        mtst_episodes = 2
        max_epochs = 2
        results_dir = 'rdev'
    exp = f'resolution'
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
            # [ 224, 'densenet121'],
            # [ 384, 'densenet121'],
            # [ 512, 'densenet121'],
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
            net_weights=net_weights,
            method=method,
            batchbased_batch_size=48,
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_subds(
        mval_episodes=100,
        mtst_episodes=10000,
        max_epochs=150,
        backbone='mobilenetv3-large-100',
        weights='i1k',
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    batchbased_train_batches = 0
    if debug:
        batchbased_train_batches = 2
        mval_episodes = 2
        mtst_episodes = 2
        max_epochs = 2
        results_dir = 'rdev'
    exp = 'subds'
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
        (data_distro, seed) = cfg
        run = '_'.join([
            data_distro,
        ])
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            net_backbone=backbone,
            net_weights=weights,
            method='batchbased',
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_subpop(
        mval_episodes=100,
        mtst_episodes=10000,
        max_epochs=150,
        backbone='mobilenetv3-large-100',
        weights='i1k',
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    batchbased_train_batches = 0
    if debug:
        batchbased_train_batches = 2
        mval_episodes = 2
        mtst_episodes = 2
        max_epochs = 2
        results_dir = 'rdev'
    exp = 'subpop'
    cfgs = list(product(
        # data_distro
        [
            'age_center',
            'age_tails',
            'sex_female',
            'sex_male',
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
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            net_backbone=backbone,
            net_weights=weights,
            method='batchbased',
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))
