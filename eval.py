""" eval.py """

import warnings
import glob
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


def save_evaluation(df, unseen, seen, hparams):
    path = join(hparams.results_dir, hparams.exp,
                hparams.run, hparams.episodes_mtst_csv)
    if isfile(path):
        df = pd.concat([pd.read_csv(path), df],
                       axis=0, ignore_index=True)
    df = df.round(2)
    cols = df.columns[1:]
    base_cols = ['seed']
    for col_metric in ('unseen', 'seen'):
        if col_metric in cols:
            base_cols.append(col_metric)
    base_cols.append('combined')
    unseen = [clazz for clazz in unseen if clazz in cols]
    seen = [clazz for clazz in seen if clazz in cols]
    df = df[base_cols + unseen + seen]
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

    if not hparams.checkpoint_name:
        ckpt_pattern = join(get_run_dir(hparams), 'checkpoints', '*.ckpt')
        best_model_path = glob.glob(ckpt_pattern)[0]

        Method = METHODS[hparams.method]
        method = Method.load_from_checkpoint(best_model_path, strict=False)
        method.hparams.episodes_mtst_csv = hparams.episodes_mtst_csv
    else:
        checkpoints_dir = 'checkpoints'
        checkpoint_path = join(checkpoints_dir, f'{hparams.checkpoint_name}.pth')
        if not isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found {checkpoint_path}")
        run_dir = join(hparams.results_dir, hparams.exp, hparams.run)
        if isdir(run_dir):
            print(f'Evaluation dir already exists: {run_dir}')
            return

        makedirs(run_dir)
        Method = METHODS.get(hparams.method, None)
        if Method is None:
            raise ValueError(f"unknown method {hparams.method}")

        method = Method(hparams)
        method.net.backbone.load_state_dict(torch.load(checkpoint_path))

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.test(method, mtst_dl, verbose=False)

    save_evaluation(method.test_df, mtst_dl.dataset.unseen,
                    mtst_dl.dataset.seen, hparams)


if __name__ == '__main__':
    import sys
    sys.exit(eval())
