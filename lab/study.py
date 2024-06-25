"""
Experiments for general study
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, train_model


SEEDS = [0]
RESULTS_DIR = 'rstudy'
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


# TODO: determine which HPs are important
def study_method_episodebase(
        net_backbone='mobilenetv3-small-075',
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    exp = 'method_episodebase'
    cfgs = list(product(
        # ------------------
        # mtrn_net_batch_pct
        [
            0.25,
            # 0.50,
            0.75,
        ],
        # mtrn_net_lr
        [
            0.00001,
            0.0005,
        ],
        # mtrn_net_steps
        [
            1,
            5,
        ],
        # mtrn_head_batch_pct
        [
            0.25,
            # 0.50,
            0.75,
        ],
        # mtrn_head_lr
        [
            0.005,
            0.001,
        ],
        # mtrn_head_steps
        [
            25,
            100,
        ],
        # ------------------
        # mtst_net_batch_pct
        [
            1.0,
        ],
        # mtst_net_lr
        [
            0.005,
        ],
        # mtst_net_steps
        [
            0,
        ],
        # mtst_head_batch_pct
        [
            0.5,
        ],
        # mtst_head_lr
        [
            0.005,
        ],
        # mtst_head_steps
        [
            100,
        ],
        # seed
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (mtrn_net_batch_pct, mtrn_net_lr, mtrn_net_steps,
         mtrn_head_batch_pct, mtrn_head_lr, mtrn_head_steps,
         mtst_net_batch_pct, mtst_net_lr, mtst_net_steps,
         mtst_head_batch_pct, mtst_head_lr, mtst_head_steps,
         seed) = cfg
        run = '_'.join([
            f'tnb-{mtrn_net_batch_pct}',
            f'tnl-{mtrn_net_lr}',
            f'tns-{mtrn_net_steps}',
            f'thb-{mtrn_head_batch_pct}',
            f'thl-{mtrn_head_lr}',
            f'ths-{mtrn_head_steps}',
            f'vnb-{mtst_net_batch_pct}',
            f'vnl-{mtst_net_lr}',
            f'vns-{mtst_net_steps}',
            f'vhb-{mtst_head_batch_pct}',
            f'vhl-{mtst_head_lr}',
            f'vhs-{mtst_head_steps}',
        ])
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS)
            results_dir = 'rdev'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            method='episodebased',
            episodebased_mtrn_net_batch_pct=mtrn_net_batch_pct,
            episodebased_mtrn_net_lr=mtrn_net_lr,
            episodebased_mtrn_net_steps=mtrn_net_steps,
            episodebased_mtrn_head_batch_pct=mtrn_head_batch_pct,
            episodebased_mtrn_head_lr=mtrn_head_lr,
            episodebased_mtrn_head_steps=mtrn_head_steps,
            episodebased_mtst_net_batch_pct=mtst_net_batch_pct,
            episodebased_mtst_net_lr=mtst_net_lr,
            episodebased_mtst_net_steps=mtst_net_steps,
            episodebased_mtst_head_batch_pct=mtst_head_batch_pct,
            episodebased_mtst_head_lr=mtst_head_lr,
            episodebased_mtst_head_steps=mtst_head_steps,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def study_method_protonet(
        net_backbone='mobilenetv3-small-075',
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):

    exp = 'method_protonet'
    cfgs = list(product(
        # protonet_encoder_type
        ['avg', 'fc'],
        # protonet_encoder_size
        [96, 128, 144],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        protonet_encoder_type, protonet_encoder_size, seed = cfg
        run = '_'.join([
            f'{protonet_encoder_type}',
            f'{protonet_encoder_size}',
        ])
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS)
            results_dir = 'rdev'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            method='protonet',
            protonet_encoder_type=protonet_encoder_type,
            protonet_encoder_size=protonet_encoder_size,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))


def study_repro(
        seeds=[0, 1, 2, 3, 4],
        results_dir=RESULTS_DIR,
        gpu=0,
        debug=False):
    exp = 'repro'
    cfgs = seeds
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        seed = cfg
        run = f'gpu{gpu}'
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))
