""" protonet.py """

import torch
import torch.optim as optim
from einops import repeat

from network import create_net
from utils import str2list
from .base import FewShotMethod, METHODS


def compute_protos(z, y):
    """Compute prototypes.

    Parameters
    ----------
    z : torch.tensor
        [b, d] tensor of `b` examples with `d` features.
    y : torch.tensor
        [b, n] tensor of `b` examples with `n` classes.

    Returns
    -------
    torch.tensor
        [n, d] tensor of `n` prototypes with `d` features.
    """
    d, n = z.shape[1], y.shape[1]

    # clone each example to match classes
    # [b, n, d] <- [b, d]
    z = repeat(z, 'b d -> b n d', n=n)

    # protos matrix
    # [b, n, d] <- [b, n]
    mask = repeat(y, 'b n -> b n d', d=d)
    # [b, n, d]
    p_matrix = z * mask

    # [n, d] <- ([n, d] <- [b, n, d]), ([n, 1] <- [b, n])
    p = p_matrix.sum(0) / y.sum(0).unsqueeze(1)

    return p


def compute_similarity(p, z):
    """Computes distance of examples to protos.

    Parameters
    ----------
    p : torch.tensor
        [n, d] tensor of `n` prototypes, `d` features.
    z : torch.tensor
        [b, d] tensor of `b` examples, `d` features.

    Returns
    -------
    torch.tensor
        (n, c) tensor of `n` examples, `c` classes.
    """
    n, b = p.shape[0], z.shape[0]

    # clone protos to match examples
    # [b, n, d] <- [n, d]
    p = repeat(p, 'n d -> b n d', b=b)

    # clone examples to match protos
    # [b, n, d] <- [b, d]
    z = repeat(z, 'b d -> b n d', n=n)

    # compute distance
    # [b, n] <- [b, n, d], [b, n, d]
    dist = torch.sqrt(torch.pow(p - z, 2).sum(dim=2))
    # convert to similarity
    # [b, n]
    sim = -(dist - dist.mean())

    return sim


@METHODS.register('protonet')
class ProtoNet(FewShotMethod):

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.convert_hparams(hparams)
        self.net = create_net(
            backbone=hparams.net_backbone,
            weights=hparams.net_weights,
            head_type=hparams.protonet_encoder_type,
            head_classes=hparams.protonet_encoder_size,
        )
        self.save_hparams(hparams, self.net)

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.protonet_lr
        )
        return opt

    def adapt_episode(self, episode):
        n_trn = episode['n_trn']
        # [b, c, h, w]
        x = episode['x']
        # [b, n]
        y_true = episode['y']

        # compute feats
        # [b, d] <- [b, c, h, w]
        z = self.net(x)

        # split episode in trn/tst
        # [b_trn, d], [b_trn, n]
        z_trn, y_true_trn = z[:n_trn], y_true[:n_trn]
        # [b_tst, d], [b_tst, n]
        z_tst, y_true_tst = z[n_trn:], y_true[n_trn:]

        # compute prototypes
        # [n, d] <- [b_trn, d], [b_trn, n]
        p = compute_protos(z_trn, y_true_trn)

        # compute logits
        # [b_tst, n] <- [n, d], [b_tst, d]
        y_lgts_tst = compute_similarity(p, z_tst)

        # compute probs
        with torch.no_grad():
            y_prob_tst = torch.sigmoid(y_lgts_tst)

        loss = self.loss_fn(y_lgts_tst, y_true_tst)

        return y_true_tst, y_prob_tst, loss

    def training_step(self, episode, episode_idx):
        y_true_tst, y_prob_tst, loss = self.adapt_episode(episode)
        self.compute_metrics_and_log('mtrn', y_true_tst, y_prob_tst,
                                     episode['seen'], episode['unseen'], loss)
        return loss

    def validation_step(self, episode, episode_idx):
        y_true_tst, y_prob_tst, loss = self.adapt_episode(episode)
        self.compute_metrics_and_log(
            'mval', y_true_tst, y_prob_tst,
            episode['seen'], episode['unseen'], loss)

    def test_step(self, episode, episode_idx):
        y_true_tst, y_prob_tst, _ = self.adapt_episode(episode)
        metrics = self.compute_full_metrics(
            y_true_tst, y_prob_tst, episode['seen'], episode['unseen'])
        self.add_episode_metrics(metrics)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--protonet_encoder_type',
                            type=str, default='avg',
                            choices=['avg', 'fc'],
                            help='protonet encoder type')
        parser.add_argument('--protonet_encoder_size',
                            type=int, default=128,
                            help='protonet encoder size')
        parser.add_argument('--protonet_lr',
                            type=float, default=0.0001,
                            help='meta-trn lr')
