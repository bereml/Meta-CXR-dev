"""
Paper Experiments
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, train_model


SEEDS = [0]
# SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = 'rdev'


def dev_episodebased(
        mtrn_episodes=1000,
        mval_episodes=100,
        mtst_episodes=10000,
        net_backbone='mobilenetv3-small-075',
        max_epochs=150,
        stop_patience=50,
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        debug=False):
    batchbased_train_batches = 0
    if debug:
        debug_steps = 1
        batchbased_train_batches = debug_steps
        mtrn_episodes = debug_steps
        mval_episodes = 1
        mtst_episodes = debug_steps
        max_epochs = debug_steps
        results_dir = 'rdev'
    exp = 'method'
    cfgs = list(product(
        #   method, precision
        [
            ['episodebased', 16],
            ['episodebased', 32],

        ],
        seeds,
    ))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (method, precision), seed = cfg
        run = '_'.join([
            method,
            f'p{precision}',
        ])
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            net_backbone=net_backbone,
            method=method,
            mtrn_episodes=mtrn_episodes,
            mval_episodes=mval_episodes,
            mtst_episodes=mtst_episodes,
            max_epochs=max_epochs,
            stop_patience=stop_patience,
            precision=precision,
            seed=seed,
        )
        aggregate_exp_df(join(results_dir, exp))
