"""
Experiments for general study
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, eval_model, train_model


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



# TODO: determine which HPs are important
def study_method_episodebase_data(
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


def studt_nf(
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
    exp = 'nf'
    cfgs = list(product(
        #   net_weights   method
        [
            ['random',    'batchbased'],
            ['random',    'protonet'],
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



def study_base_stops(
        seed=0,
        results_dir=RESULTS_DIR,
        max_epochs=150,
        debug=False):
    exp = 'base'
    cfgs = [50]
    batchbased_trn_lr = .00001
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        stop_patience = cfg
        checkpoint_name = f'base_stop_patience-{stop_patience}-{batchbased_trn_lr}'
        run = checkpoint_name
        hparams = {}
        if debug:
            hparams.update(DEBUG_HPARAMS_BB)
            results_dir = 'rdev'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            seed=seed,
            max_epochs=max_epochs,
            stop_patience=stop_patience,
            batchbased_trn_lr=batchbased_trn_lr,
            checkpoint_name=checkpoint_name,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))
