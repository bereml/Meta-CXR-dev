"""
Experiments for methods analysis
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, train_model


SEEDS = [0]
RESULTS_DIR = 'rmethod'


def method_protonet(
        mtrn_episodes=1000,
        mval_episodes=100,
        mtst_episodes=10000,
        net_backbone='mobilenetv3-small-075',
        max_epochs=150,
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
        # protonet_encoder_type
        ['avg', 'fc'],
        # protonet_encoder_size
        [96, 128, 144],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        protonet_encoder_type, protonet_encoder_size, seed = cfg
        train_model(
            results_dir=results_dir,
            exp=exp,
            net_backbone=net_backbone,
            method='protonet',
            protonet_encoder_type=protonet_encoder_type,
            protonet_encoder_size=protonet_encoder_size,
            mtrn_episodes=mtrn_episodes,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))


# TODO: update config
def method_batchbased(
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


# def method(
#         mtrn_episodes=1000,
#         mval_episodes=100,
#         mtst_episodes=10000,
#         net_backbone='mobilenetv3-large-100',
#         max_epochs=150,
#         stop_patience=50,
#         seeds=SEEDS,
#         results_dir=RESULTS_DIR,
#         debug=False):
#     batchbased_train_batches = 0
#     if debug:
#         batchbased_train_batches = 2
#         mtrn_episodes = 2
#         mval_episodes = 2
#         mtst_episodes = 2
#         max_epochs = 2
#         results_dir = 'rdev'
#     exp = 'method'
#     cfgs = list(product(
#         #   net_weights   method
#         [
#             ['random',    'batchbased'],
#             ['random',    'protonet'],
#             ['metachest', 'protonet'],
#             ['random',    'feat'],
#             ['metachest', 'feat'],
#             ['random',    'episodebased'],
#             ['metachest', 'episodebased'],
#         ],
#         seeds,
#     ))
#     for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
#         (net_weights, method), seed = cfg
#         run = '_'.join([
#             net_weights,
#             method,
#         ])
#         hparams = {}
#         if 'metachest' == net_weights:
#             net_weights = '_'.join([net_backbone, net_weights, f'seed{seed}'])
#         if method == 'batchbased':
#             hparams['batchbased_train_batches'] = batchbased_train_batches
#             # save checkpoint if not exists
#             checkpoint_name = '_'.join([net_backbone, 'metachest', f'seed{seed}'])
#             if not isfile(join('checkpoints', f'{checkpoint_name}.pth')):
#                 hparams['checkpoint_name'] = checkpoint_name
#         elif method == 'episodebased':
#             if net_weights == 'random':
#                 mtrn_episodes = 10
#                 hparams.update({
#                     'episodebased_mtrn_net_lr': 0.00001,
#                 })
#         train_model(
#             results_dir=results_dir,
#             exp=exp,
#             run=run,
#             net_backbone=net_backbone,
#             net_weights=net_weights,
#             method=method,
#             mtrn_episodes=mtrn_episodes,
#             mval_episodes=mval_episodes,
#             mtst_episodes=mtst_episodes,
#             max_epochs=max_epochs,
#             stop_patience=stop_patience,
#             seed=seed,
#             **hparams
#         )
#         aggregate_exp_df(join(results_dir, exp))