""" fomaml.py """

from copy import deepcopy

import torch
import torch.optim as optim

from network import create_net
from utils import str2bool
from .base import FewShotMethod, METHODS


@METHODS.register('fomaml')
class FOMAML(FewShotMethod):

    def __init__(self, hparams):
        super().__init__()
        hparams = self.convert_hparams(hparams)
        self.net = create_net(
            backbone=hparams.net_backbone,
            weights=hparams.net_weights,
        )
        self.save_hparams(hparams, self.net)
        self.setup_precision(self.hparams.precision)

    def setup_precision(self, precision):
        self.precision16 = precision == 16
        self.automatic_optimization = False
        if precision == 32:
            pass
        # elif precision == 16:
            # print(f"Using custom {precision}-bit precision.")
        else:
            raise ValueError(f"Invalid precision={precision}")

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.fomaml_outer_lr,
        )
        return opt

    def forward_with_loss(self, net, x, y_true):
        y_lgts = net(x)
        loss = self.loss_fn(y_lgts, y_true)
        with torch.no_grad():
            y_prob = torch.sigmoid(y_lgts)
        return y_prob, loss

    def inner_loop(self, episode, steps):
        x_trn, y_true_trn, x_tst, y_true_tst = self.split(episode)
        # lines #6-7
        net = deepcopy(self.net)
        net.new_head('fc', y_true_trn.shape[1])
        opt = optim.AdamW(net.parameters(), lr=self.hparams.fomaml_inner_lr)
        # adapt on trn
        for _ in range(steps):
            _, loss = self.forward_with_loss(net, x_trn, y_true_trn)
            opt.zero_grad()
            loss.backward()
            opt.step()
        opt.zero_grad()
        # eval on tst
        y_prob_tst, loss = self.forward_with_loss(net, x_tst, y_true_tst)
        return y_true_tst, y_prob_tst, loss, net

    def outer_loop(self, episode, episode_idx):
        # start outer loop
        num_outer_episodes = self.hparams.fomaml_outer_episodes
        if episode_idx % num_outer_episodes == 0:
            # manual zero grad to avoid initial None grads
            for p in self.net.backbone.parameters():
                p.grad = torch.zeros_like(p)
        # run inner loop step
        y_true_tst, y_prob_tst, loss, net = self.inner_loop(
            episode, self.hparams.fomaml_inner_mtrn_steps)
        # inner backward w.r.t. the test episode
        self.manual_backward(loss)
        # accumulate inner grads in the outer backbone grads
        for outer_p, inner_p in zip(self.net.backbone.parameters(),
                                    net.backbone.parameters()):
            outer_p.grad += inner_p.grad
        # end outer loop
        if (episode_idx + 1) % num_outer_episodes == 0:
            if self.hparams.fomaml_avg_grads:
                for p in self.net.backbone.parameters():
                    p.grad /= num_outer_episodes
            self.optimizers().step()
        return y_true_tst, y_prob_tst, loss

    def training_step(self, episode, episode_idx):
        # for at line #4
        y_true_tst, y_prob_tst, loss = self.outer_loop(
            episode, episode_idx)
        self.advance_global_step()
        self.compute_metrics_and_log(
            'mtrn', y_true_tst, y_prob_tst,
            episode['unseen'], episode['seen'], loss)

    @torch.enable_grad()
    def validation_step(self, episode, _):
        y_true_tst, y_prob_tst, loss, _ = self.inner_loop(
            episode, self.hparams.fomaml_inner_mtst_steps)
        self.compute_metrics_and_log(
            'mval', y_true_tst, y_prob_tst,
            episode['unseen'], episode['seen'], loss)

    @torch.enable_grad()
    def test_step(self, episode, _):
        y_true_tst, y_prob_tst, _, _ = self.inner_loop(
            episode, self.hparams.fomaml_inner_mtst_steps)
        metrics = self.compute_full_metrics(
            y_true_tst, y_prob_tst, episode['unseen'], episode['seen'])
        self.add_episode_metrics(metrics)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--fomaml_outer_lr',
                            type=float, default=0.0001,
                            help='outer meta-trn lr')
        parser.add_argument('--fomaml_outer_episodes',
                            type=int, default=2,
                            help='number of episodes in the outer loop')
        parser.add_argument('--fomaml_inner_lr',
                            type=float, default=0.0001,
                            help='inner lr')
        parser.add_argument('--fomaml_inner_mtrn_steps',
                            type=int, default=1,
                            help='steps in the inner loop')
        parser.add_argument('--fomaml_avg_grads',
                            type=str2bool, default=True,
                            help='average grans across episodes')
        parser.add_argument('--fomaml_inner_mtst_steps',
                            type=int, default=50,
                            help='steps in the inner loop')
