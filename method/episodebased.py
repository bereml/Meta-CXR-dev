""" episodebased.py """

from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.optim as optim
from torch.amp import GradScaler

from network import create_net
from .base import FewShotMethod, METHODS


@METHODS.register('episodebased')
class EpisodeBased(FewShotMethod):

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.convert_hparams(hparams)
        self.net = create_net(
            backbone=hparams.net_backbone,
            weights=hparams.net_weights,
        )
        self.save_hparams(hparams, self.net)
        self.automatic_optimization = False

    def _mset_hparams(self, mset):
        return SimpleNamespace(**{
            'net_batch_pct': getattr(self.hparams, f'episodebased_{mset}_net_batch_pct'),
            'net_lr': getattr(self.hparams, f'episodebased_{mset}_net_lr'),
            'net_steps': getattr(self.hparams, f'episodebased_{mset}_net_steps'),
            'head_batch_pct': getattr(self.hparams, f'episodebased_{mset}_head_batch_pct'),
            'head_lr': getattr(self.hparams, f'episodebased_{mset}_head_lr'),
            'head_steps': getattr(self.hparams, f'episodebased_{mset}_head_steps'),
        })

    def on_train_start(self):
        super().on_train_start()
        self.mtrn_hparams = self._mset_hparams('mtrn')

    def on_validation_start(self):
        super().on_train_start()
        self.mval_hparams = self._mset_hparams('mtst')

    def on_test_start(self):
        super().on_train_start()
        self.mtst_hparams = self._mset_hparams('mtst')


    def adapt_inner(self, x, y_true, net, opt, steps, batch_size):
        scaler = GradScaler()
        for _ in range(steps):
            idx = torch.randperm(x.shape[0])[:batch_size]
            x_batch, y_true_batch = x[idx], y_true[idx]
            with torch.autocast(self.device.type, self.float_type):
                y_lgts_batch = net(x_batch)
                loss = self.loss_fn(y_lgts_batch, y_true_batch)
            opt.zero_grad()
            loss = scaler.scale(loss)
            # got same results loss.backward() or self.manual_backward(loss)
            loss.backward()
            scaler.step(opt)
            scaler.update()

    def adapt_episode(self, episode, hparams, mtrn):
        # prepare data & model
        x_trn, y_true_trn, x_tst, y_true_tst = self.split(episode)
        n_examples, n_classes = y_true_trn.shape
        net = self.net if mtrn else deepcopy(self.net)
        net.new_head('fc', n_classes)

        # adapt full net
        batch_size = int(n_examples * hparams.net_batch_pct)
        net.unfreeze_and_train_backbone()
        net.unfreeze_and_train_head()
        opt = optim.AdamW(net.parameters(), lr=hparams.net_lr)
        self.adapt_inner(
            x_trn, y_true_trn, net, opt,
            hparams.net_steps, batch_size)

        # adapt head only
        batch_size = int(n_examples * hparams.head_batch_pct)
        net.freeze_and_eval_backbone()
        opt = optim.AdamW(net.head.parameters(), lr=hparams.head_lr)
        self.adapt_inner(
            x_trn, y_true_trn, net, opt,
            hparams.head_steps, batch_size)

        # evaluation
        net.eval()
        with torch.no_grad():
            y_lgts_tst = net(x_tst)
            y_prob_tst = torch.sigmoid(y_lgts_tst)
            loss = self.loss_fn(y_lgts_tst, y_true_tst)

        return y_true_tst, y_prob_tst, loss

    def training_step(self, episode, _):
        y_true_tst, y_prob_tst, loss = self.adapt_episode(
            episode, self.mtrn_hparams, True)
        # manually advance global_step since we are using local optimizer
        self.advance_global_step()
        self.compute_metrics_and_log(
            'mtrn', y_true_tst, y_prob_tst,
            episode['seen'], episode['unseen'], loss)

    @torch.enable_grad()
    def validation_step(self, episode, _):
        y_true_tst, y_prob_tst, loss = self.adapt_episode(
            episode, self.mval_hparams, False)
        self.compute_metrics_and_log(
            'mval', y_true_tst, y_prob_tst,
            episode['seen'], episode['unseen'], loss)

    @torch.enable_grad()
    def test_step(self, episode, _):
        y_true_tst, y_prob_tst, _ = self.adapt_episode(
            episode, self.mtst_hparams, False)
        metrics = self.compute_full_metrics(
            y_true_tst, y_prob_tst, episode['seen'], episode['unseen'], True)
        self.add_episode_metrics(metrics)

    @staticmethod
    def add_args(parser):
        # mtrn
        parser.add_argument('--episodebased_mtrn_net_batch_pct',
                            type=float, default=0.50,
                            help='data batch percentage used for inner step')
        parser.add_argument('--episodebased_mtrn_net_lr',
                            type=float, default=0.0001,
                            help='learning rate')
        parser.add_argument('--episodebased_mtrn_net_steps',
                            type=int, default=100,
                            help='number of inner training steps')
        parser.add_argument('--episodebased_mtrn_head_batch_pct',
                            type=float, default=0.75,
                            help='meta-trn data batch percentage used for inner step')
        parser.add_argument('--episodebased_mtrn_head_lr',
                            type=float, default=0.005,
                            help='meta-trn head learning rate')
        parser.add_argument('--episodebased_mtrn_head_steps',
                            type=int, default=100,
                            help='number of inner training steps')
        # mtst
        parser.add_argument('--episodebased_mtst_net_batch_pct',
                            type=float, default=1.00,
                            help='data batch percentage used for inner step')
        parser.add_argument('--episodebased_mtst_net_lr',
                            type=float, default=0.005,
                            help='learning rate')
        parser.add_argument('--episodebased_mtst_net_steps',
                            type=int, default=0,
                            help='number of inner training steps')
        parser.add_argument('--episodebased_mtst_head_batch_pct',
                            type=float, default=0.50,
                            help='data batch percentage used for inner step')
        parser.add_argument('--episodebased_mtst_head_lr',
                            type=float, default=0.005,
                            help='learning rate')
        parser.add_argument('--episodebased_mtst_head_steps',
                            type=int, default=100,
                            help='number of inner training steps')
