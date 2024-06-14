"""
Experiments for Semester 8
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, train_model


SEEDS = [0]
RESULTS_DIR = 'rsem8'


def sem8_method(
        mtrn_episodes=1000,
        mval_episodes=100,
        mtst_episodes=10000,
        net_backbone='mobilenetv3-large-100',
        max_epochs=150,
        stop_patience=50,
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    batchbased_train_batches = 0
    if debug:
        batchbased_train_batches = 2
        mtrn_episodes = 2
        mval_episodes = 2
        mtst_episodes = 2
        max_epochs = 2
        results_dir = 'rdev'
    exp = 'method'
    cfgs = list(product(
        #   net_weights   method
        [
            ['random',    'batchbased'],
            ['random',    'protonet'],
            ['metachest', 'protonet'],
            ['random',    'feat'],
            ['metachest', 'feat'],
            ['random',    'episodebased'],
            ['metachest', 'episodebased'],
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (net_weights, method), seed = cfg
        run = '_'.join([
            net_weights,
            method,
        ])
        hparams = {}
        if 'metachest' == net_weights:
            net_weights = '_'.join([net_backbone, net_weights, f'seed{seed}'])
        if method == 'batchbased':
            hparams['batchbased_train_batches'] = batchbased_train_batches
            # save checkpoint if not exists
            checkpoint_name = '_'.join([net_backbone, 'metachest', f'seed{seed}'])
            if not isfile(join('checkpoints', f'{checkpoint_name}.pth')):
                hparams['checkpoint_name'] = checkpoint_name
        elif method == 'episodebased':
            if net_weights == 'random':
                mtrn_episodes = 10
                hparams.update({
                    'episodebased_mtrn_net_lr': 0.00001,
                })
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            net_weights=net_weights,
            method=method,
            mtrn_episodes=mtrn_episodes,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            stop_patience=stop_patience,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def sem8_pretraining(
        mtrn_episodes=500,
        mval_episodes=100,
        mtst_episodes=10000,
        net_backbone='mobilenetv3-large-100',
        max_epochs=150,
        stop_patience=25,
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    batchbased_train_batches = 0
    if debug:
        batchbased_train_batches = 2
        mtrn_episodes = 2
        mval_episodes = 2
        mtst_episodes = 2
        max_epochs = 2
        results_dir = 'rdev'
    exp = 'pretraining'
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
        if 'metachest' in net_weights:
            net_weights = '_'.join([net_backbone, net_weights, f'seed{seed}'])
        if checkpoint_name:
            checkpoint_name = '_'.join([net_backbone, checkpoint_name, f'seed{seed}'])
            hparams.update({'checkpoint_name': checkpoint_name})

        if method == 'batchbased':
            hparams.update({
                'batchbased_train_batches': batchbased_train_batches
            })
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            net_weights=net_weights,
            method=method,
            mtrn_episodes=mtrn_episodes,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            stop_patience=stop_patience,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def sem8_subpop(
        mval_episodes=100,
        mtst_episodes=10000,
        backbone='mobilenetv3-large-100',
        weights='i1k',
        max_epochs=20,
        stop_patience=5,
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
        # data_config
        [
            'male',
            'female',
            'ap',
            'pa',
            'center',
            'tails',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (data_config, seed) = cfg
        run = '_'.join([
            data_config,
        ])
        data_config = join('subpop', data_config)
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_config=data_config,
            net_backbone=backbone,
            net_weights=weights,
            method='batchbased',
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            stop_patience=stop_patience,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


def sem8_subds(
        mval_episodes=100,
        mtst_episodes=10000,
        backbone='mobilenetv3-large-100',
        weights='i1k',
        max_epochs=20,
        stop_patience=5,
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
        # data_config
        [
            'chex',
            'mimic',
            'nih',
            'padchest',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (data_config, seed) = cfg
        run = '_'.join([
            data_config,
        ])
        data_config = join('subds', data_config)
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_config=data_config,
            net_backbone=backbone,
            net_weights=weights,
            method='batchbased',
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            stop_patience=stop_patience,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


def sem8_nway_unseen(
        mval_episodes=100,
        mtst_episodes=10000,
        net_backbone='mobilenetv3-large-100',
        net_weights='i1k',
        max_epochs=20,
        stop_patience=5,
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
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            net_weights=net_weights,
            method='batchbased',
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            mval_n_way=n_way,
            mval_n_unseen=n_unseen,
            mtst_n_way=n_way,
            mtst_n_unseen=n_unseen,
            max_epochs=max_epochs,
            stop_patience=stop_patience,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


def sem8_arch(
        mval_episodes=100,
        mtst_episodes=1000,
        weights='i1k',
        method='batchbased',
        batchbased_mval_lr=0.005,
        batchbased_inner_steps=100,
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
            mtrn_batch_size=48,
            batchbased_mval_lr=batchbased_mval_lr,
            batchbased_inner_steps=batchbased_inner_steps,
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


def sem8_resolution(
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
            mtrn_batch_size=48,
            batchbased_train_batches=batchbased_train_batches,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))
