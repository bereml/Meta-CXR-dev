""" eval.py """

import glob
import yaml
import warnings
from os import makedirs
from os.path import isdir, isfile, join

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from args import parse_args
from data import build_mdl
from method import METHODS
from utils import get_run_dir


def save_evaluation(df, seen, unseen, hparams):
    path = join(hparams.results_dir, hparams.exp,
                hparams.run, hparams.episodes_mtst_csv)
    if isfile(path):
        df = pd.concat([pd.read_csv(path), df],
                       axis=0, ignore_index=True)
    df = df.round(2)
    cols = df.columns[1:]
    base_cols = ['seed', 'seen', 'unseen', 'hm']
    seen = [clazz for clazz in seen if clazz in cols]
    unseen = [clazz for clazz in unseen if clazz in cols]
    df = df[base_cols + seen + unseen]
    df.to_csv(path, index=False)
    print(f'Episodes results: {path}')


def eval(run=None):
    print('==================================\n'
          '=========== EVALUATION ===========')

    hparams = parse_args()
    # replace run when is called by train()
    if run is not None:
        hparams.run = run

    seed_everything(hparams.seed, workers=True)

    if run:
    # if not hparams.checkpoint_name:
        ckpt_pattern = join(get_run_dir(hparams), 'checkpoints', '*.ckpt')
        best_model_path = glob.glob(ckpt_pattern)[0]

        Method = METHODS[hparams.method]
        method = Method.load_from_checkpoint(best_model_path, strict=False)
        method.hparams.episodes_mtst_csv = hparams.episodes_mtst_csv

    else:
        run_dir = join(hparams.results_dir, hparams.exp, hparams.run)
        if isdir(run_dir):
            print(f'Evaluation dir already exists: {run_dir}')
            return

        checkpoint_path = join('checkpoints', f'{hparams.checkpoint_name}.pth')
        if not isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found {checkpoint_path}")

        Method = METHODS.get(hparams.method, None)
        if Method is None:
            raise ValueError(f"unknown method {hparams.method}")

        method = Method(hparams)
        method.net.backbone.load_state_dict(torch.load(checkpoint_path))
        # TODO: analyze why this modfy the behaviour
        hparams.norm = method.hparams.norm
        # TODO: remove if works
        hparams.norm = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}

        makedirs(run_dir)
        with open(join(run_dir, 'hparams.yml'), 'w') as f:
            yaml.dump(vars(hparams), f, default_flow_style=False)

    hparams.norm = method.hparams.norm

    mtst_dl = build_mdl(
        'mtst',
        hparams.mtst_episodes,
        hparams.mtst_n_way,
        hparams.mtst_n_unseen,
        hparams.mtst_trn_k_shot,
        hparams.mtst_tst_k_shot,
        hparams
    )

    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        benchmark=hparams.benchmark,
        deterministic=hparams.deterministic,
        devices=hparams.devices,
        logger=False,
        inference_mode=False,
    )
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     trainer.test(method, mtst_dl, verbose=False)
    trainer.test(method, mtst_dl, verbose=False)

    save_evaluation(
        method.test_df, mtst_dl.dataset.seen, mtst_dl.dataset.unseen, hparams)


if __name__ == '__main__':
    import sys
    sys.exit(eval())
