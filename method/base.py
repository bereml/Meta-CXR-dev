""" base.py """

import warnings
from argparse import Namespace

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics.functional.classification import binary_auroc, multilabel_auroc


class Registry(dict):

    def register(self, name):
        def decorator_register(obj):
            self[name] = obj
            return obj
        return decorator_register


METHODS = Registry()


def auroc(y_prob, y_true, average):
    n_classes = y_true.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if n_classes == 1:
            return binary_auroc(y_prob.view(-1), y_true.view(-1))
        else:
            return multilabel_auroc(y_prob, y_true, n_classes, average)


@torch.inference_mode(True)
def compute_track_metrics(y_true, y_prob, unseen, seen):
    n_unseen, n_seen = len(unseen), len(seen)
    y_true = y_true.int()
    metrics = {}
    metrics['combined'] = auroc(y_prob, y_true, 'micro').item() * 100
    if n_unseen and n_seen:
        metrics['unseen'] = auroc(
            y_prob[:, :n_unseen], y_true[:, :n_unseen], 'micro').item() * 100
        metrics['seen'] = auroc(
            y_prob[:, n_unseen:], y_true[:, n_unseen:], 'micro').item() * 100
    elif n_unseen:
        metrics['unseen'] = metrics['combined']
        metrics['seen'] = ''
    else:
        metrics['unseen'] = ''
        metrics['seen'] = metrics['combined']
    return metrics


@torch.inference_mode(True)
def compute_full_metrics(y_true, y_prob, unseen, seen):
    n_unseen, n_seen = len(unseen), len(seen)
    y_true = y_true.int()
    metrics = {}
    metrics['combined'] = auroc(y_prob, y_true, 'micro').item() * 100
    if n_unseen and n_seen:
        metrics['unseen'] = auroc(
            y_prob[:, :n_unseen], y_true[:, :n_unseen], 'micro').item() * 100
        metrics['seen'] = auroc(
            y_prob[:, n_unseen:], y_true[:, n_unseen:], 'micro').item() * 100
    elif n_unseen:
        metrics['unseen'] = metrics['combined']
        metrics['seen'] = ''
    else:
        metrics['unseen'] = ''
        metrics['seen'] = metrics['combined']
    metrics.update(zip(unseen + seen,
                       auroc(y_prob, y_true, 'none').cpu().numpy() * 100))
    return metrics


class FewShotMethod(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.loss_fn = self.build_loss()
        self.episodes_metrics = []

    def convert_hparams(self, hparams):
        if isinstance(hparams, dict):
            return Namespace(**hparams)
        return hparams

    def save_hparams(self, hparams, net):
        cfg = net.backbone.pretrained_cfg
        hparams.norm = {'mean': list(cfg['mean']), 'std': list(cfg['std'])}
        self.save_hyperparameters(hparams)

    def build_loss(self):
        return nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        return None

    def on_test_epoch_end(self):
        episodes_dfs = []
        for episode_metrics in self.episodes_metrics:
            metrics = {'seed': self.hparams.seed}
            metrics.update(episode_metrics)
            episode_df = pd.Series(metrics).to_frame().T
            episodes_dfs.append(episode_df)
        df = pd.concat(episodes_dfs, axis=0, ignore_index=True)
        df['seed'] = df['seed'].astype(int)
        self.test_df = df

    def log_metrics(self, meta_set, metrics):
        metrics = {f'{k}/{meta_set}': v
                   for k, v, in metrics.items()
                   if isinstance(v, float)}
        metrics[f'loss/{meta_set}'] *= 100
        self.log_dict(metrics, on_epoch=True,
                      on_step=self.hparams['log_on_step'])

    def compute_metrics_and_log(self, meta_set, y_true, y_prob,
                                unseen, seen, loss):
        metrics = compute_track_metrics(y_true, y_prob, unseen, seen)
        metrics['loss'] = loss.item()
        self.log_metrics(meta_set, metrics)

    def compute_full_metrics(self, y_true_tst, y_prob_tst, unseen, seen):
        return compute_full_metrics(y_true_tst, y_prob_tst, unseen, seen)

    def add_episode_metrics(self, metrics):
        self.episodes_metrics.append(metrics)

    def split(self, episode):
        n_trn = episode['n_trn']
        # (n, 1, h, w)
        x = episode['x']
        # (n, c)
        y_true = episode['y']
        # split episode into trn/tst
        x_trn, y_true_trn = x[:n_trn], y_true[:n_trn]
        x_tst, y_true_tst = x[n_trn:], y_true[n_trn:]
        return x_trn, y_true_trn, x_tst, y_true_tst

    def advance_global_step(self):
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()
