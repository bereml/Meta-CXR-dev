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


def paper_arch_pn(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = f'arch_pn'
    method = 'protonet'
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
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=backbone,
            method=method,
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
        [1, 5, 15, 30],
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


def paper_method(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'method'
    cfgs = list(product(
        #   method
        [
            'batchbased',
            'protonet',
            'episodebased',
        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        method, seed = cfg
        run = '_'.join([
            method,
        ])
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'

        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            method=method,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_pretraining(
        seeds=SEEDS,
        checkpoints_dir='checkpoints',
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'pretraining'
    net_backbone = 'mobilenetv3-large-100'
    cfgs = list(product(
        #   [net_weights,           method]
        [
            ['random',              'batchbased'],
            ['i1k',                 'batchbased'],
            ['i21k',                'batchbased'],
            ['random',              'protonet'],
            ['i1k',                 'protonet'],
            ['i21k',                'protonet'],
            ['random+batchbased',   'protonet'],
            ['i1k+batchbased',      'protonet'],
            ['i21k+batchbased',     'protonet'],
            ['random+protonet',     'batchbased'],
            ['i1k+protonet',        'batchbased'],
            ['i21k+protonet',       'batchbased'],
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
        if debug:
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'

        checkpoint_name = f'{net_backbone}_{net_weights}+{method}.pth'
        if net_weights not in {'random', 'i1k', 'i21k'} :
            net_weights = f'{net_backbone}_{net_weights}.pth'

        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_weights=net_weights,
            method=method,
            checkpoints_dir=checkpoints_dir,
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


def paper_resolution_pn(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'resolution_pn'
    method = 'protonet'
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
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            image_size=image_size,
            net_backbone=net_backbone,
            method=method,
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


def paper_shift_ds_pn(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-large-100_i1k+protonet.pth',
        debug=False):
    exp = 'shift_ds_pn'
    method= 'protonet'
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
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            method=method,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))



def paper_shift_pop(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-large-100_i1k+batchbased_seed0',
        debug=False):
    exp = 'shift_pop'
    cfgs = list(product(
        # data_distro
        [
            'age_decade2',
            'age_decade3',
            'age_decade4',
            'age_decade5',
            'age_decade6',
            'age_decade7',
            'age_decade8',
            # 'sex_female',
            # 'sex_male',
            # 'complete',
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


def paper_shift_pop_pn(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-large-100_i1k+protonet.pth',
        debug=False):
    exp = 'shift_pop_pn'
    method= 'protonet'
    cfgs = list(product(
        # data_distro
        [
            'age_decade2',
            'age_decade3',
            'age_decade4',
            'age_decade5',
            'age_decade6',
            'age_decade7',
            'age_decade8',
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
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            method=method,
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

def paper_shift_view_pn(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-large-100_i1k+protonet.pth',
        debug=False):
    exp = 'shift_view_pn'
    method= 'protonet'
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
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            method=method,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


###############################################################################

def paper_pretraining_small_bb(
        seeds=SEEDS,
        checkpoints_dir='checkpoints',
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'pretraining'
    net_backbone = 'mobilenetv3-small-075'
    mtrn_trn_k_shot=30
    mtrn_tst_k_shot=30
    mval_trn_k_shot=30
    mval_tst_k_shot=30
    mtst_trn_k_shot=30
    mtst_tst_k_shot=30
    cfgs = list(product(
        #   [net_weights,           method]
        [
            ['random',              'batchbased'],
            ['i1k',                 'batchbased'],
            ['i21k',                'batchbased'],
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
        if debug:
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'

        checkpoint_name = f'{net_backbone}_{net_weights}+{method}.pth'
        if net_weights not in {'random', 'i1k', 'i21k'} :
            net_weights = f'{net_backbone}_{net_weights}.pth'

        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mtrn_trn_k_shot=mtrn_trn_k_shot,
            mtrn_tst_k_shot=mtrn_tst_k_shot,
            mval_trn_k_shot=mval_trn_k_shot,
            mval_tst_k_shot=mval_tst_k_shot,
            mtst_trn_k_shot=mtst_trn_k_shot,
            mtst_tst_k_shot=mtst_tst_k_shot,
            net_weights=net_weights,
            method=method,
            checkpoints_dir=checkpoints_dir,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))

def paper_pretraining_small_pn(
        seeds=SEEDS,
        checkpoints_dir='checkpoints',
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'pretraining'
    net_backbone = 'mobilenetv3-small-075'
    mtrn_trn_k_shot=30
    mtrn_tst_k_shot=30
    mval_trn_k_shot=30
    mval_tst_k_shot=30
    mtst_trn_k_shot=30
    mtst_tst_k_shot=30
    cfgs = list(product(
        #   [net_weights,           method]
        [
            ['random',              'protonet'],
            ['i1k',                 'protonet'],
            ['i21k',                'protonet'],
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
        if debug:
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'

        checkpoint_name = f'{net_backbone}_{net_weights}+{method}.pth'
        if net_weights not in {'random', 'i1k', 'i21k'} :
            net_weights = f'{net_backbone}_{net_weights}.pth'

        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mtrn_trn_k_shot=mtrn_trn_k_shot,
            mtrn_tst_k_shot=mtrn_tst_k_shot,
            mval_trn_k_shot=mval_trn_k_shot,
            mval_tst_k_shot=mval_tst_k_shot,
            mtst_trn_k_shot=mtst_trn_k_shot,
            mtst_tst_k_shot=mtst_tst_k_shot,
            net_weights=net_weights,
            method=method,
            checkpoints_dir=checkpoints_dir,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))

def paper_task_compĺexity_bb(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-small-075_i1k+batchbased.pth',
        debug=False):
    exp = 'task_compĺexity_bb'
    method = 'batchbased'
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
        [1, 5, 15, 30],
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
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mtst_n_way=mtst_n_way,
            mtst_n_unseen=mtst_n_unseen,
            mtst_trn_k_shot=mtst_trn_k_shot,
            method=method,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_task_compĺexity_pn(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-small-075_i1k+batchbased.pth',
        debug=False):
    exp = 'task_compĺexity_pn'
    method = 'protonet'
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
        [1, 5, 15, 30],
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
            hparams.update(DEBUG_HPARAMS_BB if method == 'batchbased'
                           else DEBUG_HPARAMS)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            mtst_n_way=mtst_n_way,
            mtst_n_unseen=mtst_n_unseen,
            mtst_trn_k_shot=mtst_trn_k_shot,
            method=method,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))

def paper_shift_pop_bb(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-small-075_i1k+batchbased.pth',
        debug=False):
    exp = 'shift_pop_bb'
    mtst_trn_k_shot=30
    mtst_tst_k_shot=30
    cfgs = list(product(
        # data_distro
        [
            'age_decade2',
            'age_decade3',
            'age_decade4',
            'age_decade5',
            'age_decade6',
            'age_decade7',
            'age_decade8',
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
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            mtst_trn_k_shot=mtst_trn_k_shot,
            mtst_tst_k_shot=mtst_tst_k_shot,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def paper_shift_ds_bb(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        checkpoint_name='mobilenetv3-small-075_i1k+batchbased.pth',
        debug=False):
    exp = 'shift_ds_bb'
    mtst_trn_k_shot=30
    mtst_tst_k_shot=30
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
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
        eval_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            data_distro=data_distro,
            mtst_trn_k_shot=mtst_trn_k_shot,
            mtst_tst_k_shot=mtst_tst_k_shot,
            seed=seed,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))
