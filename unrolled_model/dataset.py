import argparse

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def _load_dataset_file(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LocalERGraphSignalDataset(Dataset):
    """Torch Dataset for files produced by ``generate_dataset.py``.

    Each item contains noisy observations, clean signals, and the underlying
    graph.  Signal tensors keep public model shape ``(n_data, num_nodes)``;
    graph tensors keep ``(num_heads_or_1, num_nodes, k)``.
    """

    def __init__(self, path, map_location=None):
        self.path = path
        self.data = _load_dataset_file(path, map_location=map_location)
        if not isinstance(self.data, list):
            raise ValueError("dataset file must contain a list of sample dictionaries.")
        if len(self.data) == 0:
            raise ValueError("dataset file is empty.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def _stack_values(values):
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)
    return values


def local_er_collate(batch):
    """Collate list[dict] samples while preserving non-tensor metadata as lists."""
    keys = batch[0].keys()
    return {key: _stack_values([sample[key] for sample in batch]) for key in keys}


def make_local_er_dataloader(
    path,
    batch_size,
    shuffle=True,
    num_workers=0,
    map_location=None,
    drop_last=False,
):
    dataset = LocalERGraphSignalDataset(path, map_location=map_location)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=local_er_collate,
    )
    return loader


def split_local_er_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0):
    if len(dataset) < 3:
        raise ValueError("dataset must contain at least 3 samples for train/val/test splits.")
    if min(train_ratio, val_ratio, test_ratio) <= 0.0:
        raise ValueError("train/val/test ratios must all be positive.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0.0:
        raise ValueError("sum of split ratios must be positive.")

    num_samples = len(dataset)
    train_size = int(num_samples * train_ratio / total_ratio)
    val_size = int(num_samples * val_ratio / total_ratio)
    test_size = num_samples - train_size - val_size

    if train_size == 0 or val_size == 0 or test_size == 0:
        raise ValueError(
            "split ratios produce an empty split; increase dataset_size or adjust ratios."
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    train_end = train_size
    val_end = train_size + val_size
    splits = {
        "train": Subset(dataset, permutation[:train_end]),
        "val": Subset(dataset, permutation[train_end:val_end]),
        "test": Subset(dataset, permutation[val_end:]),
    }
    return splits


def make_local_er_split_loaders(
    path,
    batch_size,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=0,
    num_workers=0,
    map_location=None,
    drop_last_train=False,
):
    dataset = LocalERGraphSignalDataset(path, map_location=map_location)
    splits = split_local_er_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    loaders = {
        "train": DataLoader(
            splits["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last_train,
            collate_fn=local_er_collate,
        ),
        "val": DataLoader(
            splits["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=local_er_collate,
        ),
        "test": DataLoader(
            splits["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=local_er_collate,
        ),
    }
    return loaders, splits


def _parse_args():
    parser = argparse.ArgumentParser(description="Inspect a generated local ER dataset")
    parser.add_argument("path", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--shuffle", type=int, choices=[0, 1], default=0)
    parser.add_argument("--split", type=int, choices=[0, 1], default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.split:
        loaders, splits = make_local_er_split_loaders(
            args.path,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
        )
        print("split sizes:", {name: len(split) for name, split in splits.items()})
        for split_name, loader in loaders.items():
            batch = next(iter(loader))
            print(f"{split_name} batch keys:", sorted(batch.keys()))
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"{split_name}/{key}: {tuple(value.shape)} {value.dtype}")
                else:
                    print(f"{split_name}/{key}: {type(value).__name__} len={len(value)}")
    else:
        loader = make_local_er_dataloader(
            args.path,
            batch_size=args.batch_size,
            shuffle=bool(args.shuffle),
        )
        batch = next(iter(loader))
        print("num samples:", len(loader.dataset))
        print("batch keys:", sorted(batch.keys()))
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"{key}: {tuple(value.shape)} {value.dtype}")
            else:
                print(f"{key}: {type(value).__name__} len={len(value)}")


if __name__ == "__main__":
    main()
