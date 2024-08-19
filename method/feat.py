""" feat.py """

import torch
import torch.nn as nn
import torch.optim as optim
from einops import repeat

from network import create_net
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
    # [b, n, d]
    p_matrix = torch.zeros_like(z)
    # labels mask
    mask = y.type(torch.bool)
    # replace examples where 1 in mask (y)
    p_matrix[mask] = z[mask]

    # protos sum
    # [n, d] <- [b, n, d]
    p_sum = p_matrix.sum(0)

    # protos divisor per class
    # [n] <- [b, n]
    p_div = y.sum(0)
    # [n, d] <- [n]
    p_div = repeat(p_div, 'n -> n d', d=d)

    # protos
    # [n, d] <- [n, d], [n, d]
    p = p_sum / p_div

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
    # comvert to similarity
    # [b, n]
    sim = -(dist - dist.mean())

    return sim


@METHODS.register('feat')
class FEAT(FewShotMethod):

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.convert_hparams(hparams)
        self.net = create_net(
            backbone=hparams.net_backbone,
            weights=hparams.net_weights,
            head_type=hparams.feat_encoder_type,
            head_classes=hparams.feat_encoder_size,
        )
        self.tsfm = nn.MultiheadAttention(
            embed_dim=hparams.feat_encoder_size,
            num_heads=1,
            batch_first=True
        )
        self.save_hparams(hparams, self.net)

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.feat_lr
        )
        return opt

    def adapt_episode(self, episode, mtrn):
        n_trn = episode['n_trn']
        # [b, c, h, w]
        x = episode['x']
        # [b, n]
        y_true = episode['y']

        # compute feats
        # [b, d] <- [b, c, h, w]
        z = self.net(x)

        # split episode in trn/tst
        # [btrn, d], [btrn, n]
        z_trn, y_true_trn = z[:n_trn], y_true[:n_trn]
        # [btst, d], [btst, n]
        z_tst, y_true_tst = z[n_trn:], y_true[n_trn:]

        # compute prototypes
        # [n, d] <- [btrn, d], [btrt, n]
        p = compute_protos(z_trn, y_true_trn)

        # transform protos
        # [1, n, d] <- [n, d]
        p_batch = p.unsqueeze(0)
        # [1, n, d]
        p_batch = self.tsfm(p_batch, p_batch, p_batch, need_weights=False)[0]
        #  [n, d] <- [1, n, d]
        p = p_batch.squeeze()

        # compute logits
        # [btst, n] <- [n, d], [btst, d]
        y_lgts_tst = compute_similarity(p, z_tst)

        # 1 <- [btst, n], [btst, n]
        loss = self.loss_fn(y_lgts_tst, y_true_tst)

        # compute probs
        with torch.no_grad():
            y_prob_tst = torch.sigmoid(y_lgts_tst)

        if mtrn:
            n = y_true.shape[1]
            b, d = z.shape
            # [n, b, d] <- [b, d]
            aux_task = repeat(z, 'b d -> n b d', n=n)
            # mask
            # [n, b] <- [b, n]
            attn_mask = y_true.T
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0)
            # [n, b, b] <- [n, b]
            attn_mask = repeat(attn_mask, 'n b -> n v b', v=b)

            # [n, b, d]
            aux_emb = self.tsfm(aux_task, aux_task, aux_task,
                                need_weights=False, attn_mask=attn_mask)[0]
            # [n, b, d] <- [b, n]
            mask = repeat(y_true, 'b n -> n b d', d=d)
            # [n, b, d]
            aux_center = aux_emb * mask
            # [n, d] <- [n, b, d], [b, n]
            aux_center = aux_center.sum(dim=1) / y_true.sum(0).unsqueeze(1)

            # [b, n] <- [n, d], [b, d]
            y_lgts = compute_similarity(aux_center, z)
            # 1 <- [b, n], [b, n]
            loss_reg = self.loss_fn(y_lgts, y_true)
            loss = loss + loss_reg

        return y_true_tst, y_prob_tst, loss

    def training_step(self, episode, episode_idx):
        y_true_tst, y_prob_tst, loss = self.adapt_episode(episode, True)
        self.compute_metrics_and_log('mtrn', y_true_tst, y_prob_tst,
                                     episode['unseen'], episode['seen'], loss)
        return loss

    def validation_step(self, episode, episode_idx):
        y_true_tst, y_prob_tst, loss = self.adapt_episode(episode, False)
        self.compute_metrics_and_log(
            'mval', y_true_tst, y_prob_tst,
            episode['unseen'], episode['seen'], loss)

    def test_step(self, episode, episode_idx):
        y_true_tst, y_prob_tst, _ = self.adapt_episode(episode, False)
        metrics = self.compute_full_metrics(
            y_true_tst, y_prob_tst, episode['unseen'], episode['seen'])
        self.add_episode_metrics(metrics)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--feat_encoder_type',
                            type=str, default='avg',
                            choices=['avg', 'fc'],
                            help='feat encoder type')
        parser.add_argument('--feat_encoder_size',
                            type=int, default=64,
                            help='feat encoder size')
        parser.add_argument('--feat_lr',
                            type=float, default=0.0005,
                            help='mtrn lr')
