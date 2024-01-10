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

from utils import load_config, read_toml


TRN_IDX, TST_IDX = 0, 1


def _load_data(data_config, mset):
    df_path = join('dsrepo', f'metachest.csv')
    if not isfile(df_path):
        raise ValueError(f"MetaChest CSV not found {df_path}")
    filter_df_path = join('dsrepo', f'{data_config}.csv')
    if not isfile(filter_df_path):
        raise ValueError(f"Filter CSV not found {filter_df_path}")
    config_path = join('dsrepo', f'{data_config}.toml')
    if not isfile(config_path):
        raise ValueError(f"Config file not found {config_path}")

    df = pd.read_csv(df_path)
    filter_df = pd.read_csv(filter_df_path)
    mclasses = read_toml(config_path)

    if mset == 'mtrn':
        unseen, seen = mclasses['mtrn'], []
    elif mset == 'mval':
        unseen, seen = mclasses['mval'], mclasses['mtrn']
    else:
        unseen, seen = mclasses['mtst'], mclasses['mtrn']

    classes = unseen + seen
    df = df.loc[filter_df[mset] == 1, ['dataset', 'name'] + classes]
    df[classes] = df[classes].fillna(0).astype(int)

    return df, unseen, seen


class XRayDS(Dataset):
    """XRay Dataset."""

    def __init__(self, mset, tsfm, hparams):
        """
        Parameters
        ----------

        mset : {'mtrn'}
            Meta-dataset to load.
        tsfm : Callable
            Image transformation.
        hparams : SimpleNamespace
            data_config : str
                Data configuration name.
            image_size : int
                Image size.
        """

        if mset not in {'mtrn'}:
            raise ValueError(f'invalid mset={mset}')
        if hparams.image_size not in {224, 384, 512, 768, 1024}:
            raise ValueError(f'invalid image_size={hparams.image_size}')

        metachest_dir = load_config()['datasets']['metachest']
        images_dir = join(metachest_dir, f'images-{hparams.image_size}')
        if not isdir(images_dir):
            raise ValueError(f'invalid images_dir={images_dir}')

        df, unseen, seen = _load_data(hparams.data_config, mset)
        self.mset = mset
        self.df = df
        self.unseen = unseen
        self.seen = seen
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

        example = [self.unseen, self.seen, dataset, name, x, y]
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
            data_config : str
                Data configuration name.
            image_size : int
                Image size.
        """

        if mset not in {'mtrn', 'mval', 'mtst'}:
            raise ValueError(f'invalid mset={mset}')
        if hparams.image_size not in {224, 384, 512, 768, 1024}:
            raise ValueError(f'invalid image_size={hparams.image_size}')

        metachest_dir = load_config()['datasets']['metachest']
        images_dir = join(metachest_dir, f'images-{hparams.image_size}')
        if not isdir(images_dir):
            raise ValueError(f'invalid images_dir={images_dir}')

        df, unseen, seen = _load_data(hparams.data_config, mset)
        self.mset = mset
        self.df = df
        self.unseen = unseen
        self.seen = seen
        self.images_dir = images_dir
        self.trn_tsfm = trn_tsfm
        self.tst_tsfm = tst_tsfm
        self.tsfm = [trn_tsfm, tst_tsfm]

    def __getitem__(self, example):
        """Returns the example.

        Parameters
        ----------
        example : [int, str, str, [str], [str], [int]]
            Example with [subset, dataset, name, unseen, seen, labels]

        Returns
        -------
        [int, [int], [str], str, torch.tensor, torch.tensor]
            List of [subset, unseen, seen, dataset, name, x, y].
        """
        subset, dataset, name, unseen, seen, labels = example

        path = join(self.images_dir, dataset, f'{name}.jpg')
        x = read_image(path, ImageReadMode.RGB)
        x = self.tsfm[subset](x)

        y = torch.tensor(labels, dtype=torch.float32)

        example = [subset, unseen, seen, dataset, name, x, y]
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
            A list of [subset, dataset, name, unseen, seen, labels] for each example.
        """
        # select classes
        random.shuffle(self.unseen)
        random.shuffle(self.seen)
        unseen, seen = self.unseen[:self.n_unseen], self.seen[:self.n_seen]
        classes = unseen + seen
        exclude = self.unseen[self.n_unseen:] + self.seen[self.n_seen:]


        # exclude examples and classes
        df = self.df.loc[~self.df[exclude].astype(bool).any(axis=1),
                         ['dataset', 'name'] + classes]
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
        classes = unseen + seen
        episode_df = episode_df[['set', 'dataset', 'name'] + classes]

        episode = [
            [
                example['set'],
                example['dataset'],
                example['name'],
                unseen,
                seen,
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
    for unseen, seen, dataset, name, x, y in batch:
        datasets.append(dataset)
        names.append(name)
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs)
    y = torch.stack(ys)
    episode = {
        'unseen': unseen,
        'seen': seen,
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
    for subset, unseen, seen, dataset, name, x, y in episode:
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
        data_configs_dir : str
            Parent dir of configuration directories.
        data_config_dir : str
            Data CSV configuration directory.
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
        data_config_dir : str
            Data CSV configuration directory.
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
        image_size=384, image_channels=1, data_aug=False,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        batch_size=32,
        num_workers=0,
        batches=1,
        data_config='complete',
        seed=0,
        debug=True):

    hparams = locals()

    from itertools import islice as take
    from pprint import pprint
    from types import SimpleNamespace

    hparams = SimpleNamespace(**hparams)

    dl = build_dl(mset, batch_size, hparams)

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
        print(unseen, seen)

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
        image_size=384, image_channels=1, data_aug=False,
        n_episodes=1, n_way=3, n_unseen=1,
        trn_k_shot=2, tst_k_shot=6,
        norm={'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]},
        num_workers=0,
        data_config='complete',
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
        unseen = episode['unseen']
        seen = episode['seen']
        n_trn = episode['n_trn']
        n_tst = episode['n_tst']
        dataset = episode['dataset']
        name = episode['name']
        x = episode['x']
        y = episode['y']

        print(f'n_trn={n_trn} n_tst={n_tst}')
        print(f'name shape={len(name)}')
        print(f'x shape={x.shape} dtype={x.dtype} '
              f'mean={x.type(torch.float).mean().round(decimals=2)} '
              f'min={x.min()} max={x.max()}')
        print(f'y shape={y.shape} dtype={y.dtype}')
        print(unseen, seen)

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


if __name__ == '__main__':
    import fire
    fire.Fire()
