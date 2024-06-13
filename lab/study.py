"""
Experiments for general study
"""

from itertools import product
from os.path import join, isfile

from tqdm import tqdm

from utils import aggregate_exp_df, train_model


SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = 'rstudy'


DEBUG_PARAMS = {
    'batchbased_train_batches': 1,
    'mtrn_episodes': 1,
    'mval_episodes': 1,
    'mtst_episodes': 1,
    'max_epochs': 1,
}


def study_repro(
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        gpu=0,
        debug=False):
    hparams = {}
    if debug:
        hparams.update(DEBUG_PARAMS)
        results_dir = 'rdev'
    exp = 'repro'
    cfgs = seeds
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        seed = cfg
        run = f'gpu{gpu}'
        train_model(
            results_dir=results_dir,
            exp=exp,
            run=run,
            seed=seed,
            **hparams
        )
        aggregate_exp_df(join(results_dir, exp))
