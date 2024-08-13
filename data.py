import random
from os.path import join, isdir, isfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.io import ImageReadMode, read_image

from utils import load_config


IMAGE_SIZES = {224, 336, 384, 448, 512, 768, 1024}
TRN_IDX, TST_IDX = 0, 1


def _filter_mset(mset, mclasses, df, n_metadata_cols=5):
    mval_mtst_examples = df[mclasses['mval'] + mclasses['mtst']].any(axis=1)
    if mset == 'mtrn':
        # keep examples with only mtrn classes
        mtrn_only_examples = ~mval_mtst_examples
        df = df[mtrn_only_examples]
        classes = mclasses['mtrn']
    else:
        # discarding examples with only mtrn classes
        df = df[mval_mtst_examples]
        # keep examples with mtrn+mset clases
        mtrn_mset_classes = mclasses['mtrn'] + mclasses[mset]
        mtrn_mset_examples = df[mtrn_mset_classes].any(axis=1)
        df = df[mtrn_mset_examples]
        classes = mtrn_mset_classes
    cols = list(df.columns[:n_metadata_cols]) + classes
    df = df[cols]
    return df


def _load_data(config, mset, distro):

    metachest_dir = config['metachest_dir']
    df_path = join(metachest_dir, f'metachest.csv')
    if not isfile(df_path):
        raise ValueError(f"MetaChest CSV not found {df_path}")
    df = pd.read_csv(df_path)

    if distro != 'complete':
        distro_path = join(metachest_dir, 'distro', f'{distro}.csv')
        if not isfile(distro_path):
            raise ValueError(f"Distro CSV not found {distro_path}")
        distro_mask = pd.read_csv(distro_path).iloc[:, 0].astype(bool)
        df = df[distro_mask]

    mclasses = {'mtrn': config['mtrn'],
                'mval': config['mval'],
                'mtst': config['mtst']}
    df = _filter_mset(mset, mclasses, df)

    seen, unseen = {
        #        seen              unseen
        'mtrn': ([],               mclasses['mtrn']),
        'mval': (mclasses['mtrn'], mclasses['mval']),
        'mtst': (mclasses['mtrn'], mclasses['mtst']),
    }[mset]

    seen = [clazz for clazz in seen if df[clazz].any()]
    unseen = [clazz for clazz in unseen if df[clazz].any()]

    df = df[['dataset', 'name'] + seen + unseen]
    df[seen + unseen] = df[seen + unseen].fillna(0).astype(int)

    return seen, unseen, df


class XRayDS(Dataset):
    """XRay Dataset."""

    def __init__(self, mset, tsfm, hparams):
        """
        Parameters
        ----------

        mset : {'mtrn'}
            Meta-dataset.
        tsfm : Callable
            Image transformation.
        hparams : SimpleNamespace
            data_distro : str
                Data distribution name.
            image_size : int
                Image size.
        """

        if mset not in {'mtrn'}:
            raise ValueError(f'Invalid mset={mset}')
        if hparams.image_size not in IMAGE_SIZES:
            raise ValueError(f'Invalid image_size={hparams.image_size}')

        config = load_config()
        metachest_dir = config['metachest_dir']
        images_dir = join(metachest_dir, f'images-{hparams.image_size}')
        if not isdir(images_dir):
            raise ValueError(f'Dir not found images_dir={images_dir}')

        seen, unseen, df = _load_data(config, mset, hparams.data_distro)
        self.mset = mset
        self.df = df
        self.seen = seen
        self.unseen = unseen
        self.images_dir = images_dir
        self.tsfm = tsfm
        self.samples = df[['dataset', 'name']].values.tolist()
        self.classes = df.columns[2:].tolist()
        self.labels = df[self.classes].to_numpy()

    def __getitem__(self, i):
        dataset, name = self.samples[i]
        path = join(self.images_dir, dataset, f'{name}.jpg')
        x = read_image(path, ImageReadMode.RGB)
        x = self.tsfm(x)

        y = self.labels[i]
        y = torch.tensor(y, dtype=torch.float32)

        example = [self.seen, self.unseen, dataset, name, x, y]
        return example

    def __len__(self):
        return len(self.samples)


class XRayMetaDS(Dataset):
    """XRay Meta-Dataset."""

    def __init__(self, mset, trn_tsfm, tst_tsfm, hparams):
        """
        Parameters
        ----------

        mset : {'mtrn', 'mval', 'mtst'}
            Meta-dataset to load.
        trn_tsfm : Callable
            Train transformation.
        tst_tsfm : Callable
            Test transformation.
        hparams : SimpleNamespace
            data_distro : str
                Data distribution name.
            image_size : int
                Image size.
        """

        if mset not in {'mtrn', 'mval', 'mtst'}:
            raise ValueError(f'invalid mset={mset}')
        if hparams.image_size not in IMAGE_SIZES:
            raise ValueError(f'invalid image_size={hparams.image_size}')

        config = load_config()
        metachest_dir = config['metachest_dir']
        images_dir = join(metachest_dir, f'images-{hparams.image_size}')
        if not isdir(images_dir):
            raise ValueError(f'invalid images_dir={images_dir}')

        seen, unseen, df = _load_data(config, mset, hparams.data_distro)
        self.mset = mset
        self.df = df
        self.seen = seen
        self.unseen = unseen
        self.images_dir = images_dir
        self.trn_tsfm = trn_tsfm
        self.tst_tsfm = tst_tsfm
        self.tsfm = [trn_tsfm, tst_tsfm]

    def __getitem__(self, example):
        """Returns the example.

        Parameters
        ----------
        example : [int, str, str, [str], [str], [int]]
            Example with [subset, dataset, name, seen, unseen, labels]

        Returns
        -------
        [int, [int], [str], str, torch.tensor, torch.tensor]
            List of [subset, seen, unseen, dataset, name, x, y].
        """
        subset, dataset, name, seen, unseen, labels = example

        path = join(self.images_dir, dataset, f'{name}.jpg')
        x = read_image(path, ImageReadMode.RGB)
        x = self.tsfm[subset](x)

        y = torch.tensor(labels, dtype=torch.float32)

        example = [subset, seen, unseen, dataset, name, x, y]
        return example

    def __len__(self):
        return len(self.df)


def sample_at_least_k_shot(df, k_shot):
    """Samples an episode with at least `k` examples per class.

        Parameters
        ----------
        df : pd.DataFrame
            Array of labels with first column as index.
        k_shot : int
            Numbers of k-shot examples for the episode.

        Returns
        -------
        pd.DataFrame
            Episode dataframe.
        """
    kdf = pd.DataFrame(columns=df.columns)
    for clazz in df.columns[2:].values:
        # count missing examples for the class
        k_miss = k_shot - kdf[clazz].sum()
        if k_miss > 0:
            # select the k missing examples
            cdf = df[df[clazz] == 1].iloc[:k_miss]
            # append them
            kdf = pd.concat([kdf, cdf])
            # remove them
            df.drop(cdf.index, inplace=True)
    return kdf


class EpisodeSampler(Sampler):
    """Multi-label episode sampler."""

    def __init__(self, dataset, n_episodes, n_way, n_unseen, trn_k_shot, tst_k_shot):
        """
        Parameters
        ----------
        dataset : XRayMetaDS
            The dataset.
            labels : np.ndarray
                Dataset labels.
        n_episodes : int
            Number of episodes to generate.
        n_way : int
            Number of classes for episode.
        trn_k_shot : int
            Minimal number of examples per classes for episode in training.
        tst_k_shot : int
            Minimal number of examples per classes for episode in testing.
        """
        if n_unseen > n_way:
            raise ValueError(
                f'Metaset {dataset.mset}: n_unseen={n_unseen} > n_way={n_way}')
        self.df = dataset.df
        self.seen = list(dataset.seen)
        self.unseen = list(dataset.unseen)
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.trn_k_shot = trn_k_shot
        self.tst_k_shot = tst_k_shot
        if self.seen:
            self.n_unseen = min(n_unseen, len(self.unseen))
            self.n_seen = n_way - self.n_unseen
        else:
            self.n_unseen = n_way
            self.n_seen = 0


    def generate_episode(self):
        """Generates an episode.

        Returns
        -------
        [int, str, str, [str], [str], [int]]
            A list of [subset, dataset, name, seen, unseen, labels] for each example.
        """
        # select classes
        random.shuffle(self.seen)
        random.shuffle(self.unseen)
        seen = self.seen[:self.n_seen]
        unseen = self.unseen[:self.n_unseen]
        classes = seen + unseen
        exclude_classes = self.seen[self.n_seen:] + self.unseen[self.n_unseen:]

        # exclude examples with non selected clases
        df = self.df.loc[~(self.df[exclude_classes].any(axis=1))]
        # filter columns
        df = df[['dataset', 'name'] + classes]
        df = df.reset_index(drop=True)

        # sort classes by frequency
        classes = list(df[classes].sum(axis=0).sort_values(ascending=True).index)
        df = df[['dataset', 'name'] + classes]

        # shuffle dataset
        df = df.sample(df.shape[0]).reset_index(drop=True)

        trn_df = sample_at_least_k_shot(df, self.trn_k_shot)
        tst_df = sample_at_least_k_shot(df, self.tst_k_shot)
        trn_df['set'] = trn_df.shape[0] * [TRN_IDX]
        tst_df['set'] = tst_df.shape[0] * [TST_IDX]
        episode_df = pd.concat([trn_df, tst_df])
        episode_df = episode_df[['set', 'dataset', 'name'] + classes]

        episode = [
            [
                example['set'],
                example['dataset'],
                example['name'],
                seen,
                unseen,
                [example[c] for c in classes]
            ]
            for example in episode_df.to_dict('records')
        ]

        return episode

    def __iter__(self):
        """Yields a new episode.

        Yields
        -------
        [[int, [int], int]]
            A list of subset, classes and index for each example.
        """
        for _ in range(self.n_episodes):
            episode = self.generate_episode()
            yield episode

    def __len__(self):
        return self.n_episodes


def build_tsfm(data_aug, hparams, debug):
    tsfm = []

    if data_aug:
        tsfm.extend([
            T.RandomAffine(degrees=15,
                           translate=(0.1, 0.1),
                           scale=(0.9, 1.1))
        ])
    if not debug:
        mean, std = hparams.norm['mean'], hparams.norm['std']
        tsfm.extend([
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
    tsfm = nn.Sequential(*tsfm)
    return tsfm


def collate_batch(batch):
    datasets, names, xs, ys = [], [], [], []
    for seen, unseen, dataset, name, x, y in batch:
        datasets.append(dataset)
        names.append(name)
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs)
    y = torch.stack(ys)
    episode = {
        'seen': seen,
        'unseen': unseen,
        'dataset': datasets,
        'name': names,
        'x': x,
        'y': y
    }
    return episode


def collate_episode(episode):
    # assuming TRN_IDX, TST_IDX = 0, 1
    size = [0, 0]
    datasets, names, xs, ys = [], [], [], []
    for subset, seen, unseen, dataset, name, x, y in episode:
        size[subset] += 1
        datasets.append(dataset)
        names.append(name)
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs)
    y = torch.stack(ys)
    episode = {
        'n_trn': size[TRN_IDX],
        'n_tst': size[TST_IDX],
        'unseen': unseen,
        'seen': seen,
        'dataset': datasets,
        'name': names,
        'x': x,
        'y': y
    }
    return episode


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dl(mset, batch_size, hparams):
    """Builds a meta dataloader for XRayMetaDS episode sampling.

    Parameters
    ----------
    mset : {'mtrn', 'mval', 'mtst'}
        Meta-dataset to load.
    hparams : SimpleNamespace
        data_aug : bool
            Enable data augmentation.
        num_workers : int
            Number of process for the dataloader.
        seed : bool
            Seed to init generators.
        debug : bool, default=True
            If True, prints loading info.

    Returns
    -------
    DataLoader
        The meta dataloader.
    """

    tsfm = build_tsfm(hparams.data_aug, hparams, hparams.debug)

    dataset = XRayDS(mset, tsfm, hparams)

    g = torch.Generator()
    g.manual_seed(hparams.seed)

    dl = DataLoader(
        dataset=dataset,
        collate_fn=collate_batch,
        shuffle=True,
        batch_size=batch_size,
        num_workers=hparams.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    return dl


def build_mdl(mset, n_episodes, n_way, n_unseen, trn_k_shot, tst_kshot, hparams):
    """Builds a meta dataloader for XRayMetaDS episode sampling.

    Parameters
    ----------
    mset : {'mtrn', 'mval', 'mtst'}
        Meta-dataset to load.
    n_episodes : int
        Number of episodes.
    n_way : int
        Number of classes per episode.
    n_unseen : int
        Number of unseen classes per episode.
    trn_k_shot : int
        Minimal number of examples per classes for episode in training.
    tst_k_shot : int
        Minimal number of examples per classes for episode in testing.
    hparams : SimpleNamespace
        distro : str
            Distribution name.
        num_workers : int
            Number of process for the dataloader.
        debug : bool, default=True
            If True, prints loading info.

    Returns
    -------
    DataLoader
        The meta dataloader.
    """

    trn_tsfm = build_tsfm(hparams.data_aug, hparams, hparams.debug)
    tst_tsfm = build_tsfm(False, hparams, hparams.debug)

    dataset = XRayMetaDS(mset, trn_tsfm, tst_tsfm, hparams)

    sampler = EpisodeSampler(
        dataset, n_episodes, n_way, n_unseen, trn_k_shot, tst_kshot)

    g = torch.Generator()
    g.manual_seed(hparams.seed)

    mdl = DataLoader(
        dataset, batch_sampler=sampler,
        collate_fn=collate_episode,
        num_workers=hparams.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    return mdl


def show_grid(x):
    import matplotlib.pyplot as plt
    import numpy as np

    from torchvision.utils import make_grid

    grid = make_grid(x, value_range=(0, 255))
    grid = np.array(F.to_pil_image(grid.detach()))

    plt.imshow(grid)
    plt.show()


def test_build_dl(
        mset='mtrn',
        image_size=384, data_aug=False,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        batch_size=64,
        num_workers=0,
        batches=1,
        data_distro='complete',
        seed=0,
        debug=True):

    hparams = locals()

    from itertools import islice as take
    from pprint import pprint
    from types import SimpleNamespace

    hparams = SimpleNamespace(**hparams)

    dl = build_dl(mset, batch_size, hparams)

    print(
        'Dataloader:\n'
        f'  number of examples: {len(dl.dataset)}\n'
        f'          batch size: {batch_size}\n'
        f'   number of batches: {len(dl)}\n'
    )

    for batch in take(dl, batches):
        unseen = batch['unseen']
        seen = batch['seen']
        dataset = batch['dataset']
        name = batch['name']
        x = batch['x']
        y = batch['y']

        print(f'x shape={x.shape} dtype={x.dtype} '
              f'mean={x.type(torch.float).mean().round(decimals=2)} '
              f'min={x.min()} max={x.max()}')
        print(f'y shape={y.shape} dtype={y.dtype}')
        print(seen, unseen)

        datasets = np.array(dataset)
        names = np.array(name)
        data = np.column_stack([datasets, names, y.type(torch.int)])
        cols = ['dataset', 'name'] + unseen + seen
        df = pd.DataFrame(data, columns=cols)
        pd.set_option('display.max_colwidth', 500)
        print(df)

        if debug:
            show_grid(x)


def test_build_mdl(
        mset='mtrn',
        image_size=384, data_aug=False,
        n_episodes=1, n_way=3, n_unseen=1,
        trn_k_shot=5, tst_k_shot=15,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        num_workers=0,
        data_distro='complete',
        seed=0,
        debug=True):

    hparams = locals()

    from itertools import islice as take
    from types import SimpleNamespace

    hparams = SimpleNamespace(**hparams)

    mdl = build_mdl(
        mset, n_episodes, n_way, n_unseen, trn_k_shot, tst_k_shot, hparams)

    episode_size = []

    for episode in take(mdl, n_episodes):
        seen = episode['seen']
        unseen = episode['unseen']
        n_trn = episode['n_trn']
        n_tst = episode['n_tst']
        dataset = episode['dataset']
        name = episode['name']
        x = episode['x']
        y = episode['y']

        print(f'seen={seen} unseen={unseen}')
        print(f'n_trn={n_trn} n_tst={n_tst}')
        print(f'name shape={len(name)}')
        print(f'x shape={x.shape} dtype={x.dtype} '
              f'mean={x.type(torch.float).mean().round(decimals=2)} '
              f'min={x.min()} max={x.max()}')
        print(f'y shape={y.shape} dtype={y.dtype}')

        subset = n_trn * ['trn'] + n_tst * ['tst']
        datasets = np.array(dataset)
        names = np.array(name)
        data = np.column_stack([subset, datasets, names, y.type(torch.int)])
        cols = ['subset', 'dataset', 'name'] + unseen + seen
        df = pd.DataFrame(data, columns=cols)
        pd.set_option('display.max_colwidth', 500)
        print(df)

        if debug:
            show_grid(x)

        episode_size.append(n_trn + n_tst)

    print(f'Mean episode size: {np.mean(episode_size)}')


def compute_mean_mdl(
        mset='mtrn',
        image_size=384, data_aug=False,
        n_episodes=1000, n_way=3, n_unseen=1,
        trn_k_shot=5, tst_k_shot=15,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        num_workers=0,
        data_distro='complete',
        seed=0,
        debug=True):

    hparams = locals()

    from types import SimpleNamespace

    from tqdm import tqdm


    hparams = SimpleNamespace(**hparams)

    mdl = build_mdl(
        mset, n_episodes, n_way, n_unseen, trn_k_shot, tst_k_shot, hparams)

    episode_sizes = []
    for episode in tqdm(mdl):
        episode_sizes.append(episode['n_trn'] + episode['n_tst'])

    total = len(mdl.dataset)
    episode_size = np.mean(episode_sizes)
    examples_used = np.sum(episode_sizes)


    print(
        'Dataloader:\n'
        f'      total examples: {total}\n'
        f'        episode size: {episode_size}\n'
        f'   number of batches: {len(mdl)}\n'
        f'       examples used: {examples_used}\n'
        f'     proportion used: {examples_used / total}\n'
        f'           eval freq: {total / examples_used}\n'
    )



if __name__ == '__main__':
    import fire
    fire.Fire()
