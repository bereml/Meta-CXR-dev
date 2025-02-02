"""
Experiments for Publication
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, eval_model, train_model


SEEDS = [0]
RESULTS_DIR = 'rpaper'
DEBUG_HPARAMS = {
    'mtrn_episodes': 1,
    'mval_episodes': 1,
    'mtst_episodes': 1,
    'max_epochs': 1,
}
DEBUG_HPARAMS_BB = {
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
    exp = 'base'
    run = 'base'
    cfgs = seeds
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        seed = cfg
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
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
        mtrn_batch_size=48,
        debug=False):
    exp = f'arch_batch-size-{mtrn_batch_size}'
    cfgs = list(product(
        # arch
        [
            # efficient
            'mobilenetv3-small-075',
            'mobilevitv2-050',
            'mobilenetv3-large-100',
            'convnext-atto',
            'convnextv2-atto',
            'mobilevitv2-100',
            # large
            'densenet121',
            'mobilevitv2-200',
            'convnextv2-nano',
            'densenet161',
            'convnext-tiny',
            'convnextv2-tiny',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        backbone, seed = cfg
        run = '_'.join([
            backbone,
        ])
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=backbone,
            mtrn_batch_size=mtrn_batch_size,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_gfsl(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='base',
        debug=False):
    exp = 'gfsl'
    cfgs = list(product(
        # mtst_n_way, mtst_n_unseen
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
        # mtst_trn_k_shot
        [1, 5, 15],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (mtst_n_way, mtst_n_unseen), mtst_trn_k_shot, seed = cfg
        run = '_'.join([
            f'nway-{mtst_n_way}',
            f'unseen-{mtst_n_unseen}',
            f'kshot-{mtst_trn_k_shot:02d}',
        ])
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mtst_n_way=mtst_n_way,
            mtst_n_unseen=mtst_n_unseen,
            mtst_trn_k_shot=mtst_trn_k_shot,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_pretraining(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'pretraining'
    net_backbone = 'mobilenetv3-large-100'
    cfgs = list(product(
        #   [net_weights,      method,       checkpoint_name]
        [
            ['random',         'batchbased', 'metachest'],
            ['i1k',            'batchbased', 'i1k-metachest'],
            ['i21k',           'batchbased', 'i21k-metachest'],
            ['random',         'protonet',   None],
            ['i1k',            'protonet',   None],
            ['i21k',           'protonet',   None],
            ['metachest',      'protonet',   None],
            ['i1k-metachest',  'protonet',   None],
            ['i21k-metachest', 'protonet',   None],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (net_weights, method, checkpoint_name), seed = cfg
        run = '_'.join([
            net_weights,
            method,
        ])
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'

        # include backbone name in net_weights & checkpoint_name
        if 'metachest' in net_weights:
            net_weights = '_'.join([net_backbone, net_weights, f'seed{seed}'])
        if checkpoint_name:
            checkpoint_name = '_'.join([net_backbone, checkpoint_name, f'seed{seed}'])
            hparams['checkpoint_name'] = checkpoint_name

        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_weights=net_weights,
            method=method,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_resolution(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        mtrn_batch_size=32,
        debug=False):
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
            [ 224, 'densenet121'],
            [ 384, 'densenet121'],
            [ 512, 'densenet121'],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (image_size, net_backbone), seed = cfg
        run = '_'.join([
            net_backbone,
            f'{image_size:04d}',
        ])
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
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
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
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
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
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
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
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


# -----------------------------------------------
# unfinished


# def paper_foundation(
#         seeds=SEEDS,
#         results_dir=RESULTS_DIR,
#         mtrn_batch_size=48,
#         debug=False):
#     exp = 'foundation'
#     cfgs = list(product(
#         # image_size, net_backbone, net_weights
#         [
#             # [336, 'eva02-tiny', 'i21k'],
#             # [336, 'eva02-small', 'i21k'],
#             [448, 'eva02-large', 'mim_m38m_ft_in22k_in1k'],
#             # [448, 'eva02-large', 'mim_in22k_ft_in22k_in1k'],
#         ],
#         seeds,
#     ))
#     for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
#         (image_size, net_backbone, net_weights), seed = cfg
#         run = '_'.join([
#             net_backbone,
#             net_weights
#         ])
#         hparams = {}
#         if debug:
#             hparams.update(DEBUG_HPARAMS_BB)
#             results_dir = 'rdev'
#         train_model(
#             results_dir=results_dir,
#             exp=exp,
#             run=run,
#             image_size=image_size,
#             net_backbone=net_backbone,
#             net_weights=net_weights,
#             mtrn_batch_size=mtrn_batch_size,
#             seed=seed,
#             **hparams
#         )
#         aggregate_exp_df(join(results_dir, exp))

