"""Data loaders for downstream classification and regression evaluation."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class ClassificationDataset(Dataset):
    """Dataset for supervised time-series classification/regression samples."""

    def __init__(self, path, ratio_patches=20, transform_to_patch=True, regression=False):
        self.ratio_patches = ratio_patches
        self.transform_to_patch = transform_to_patch
        self.regression = regression
        series, labels = _read_supervised_timeseries(path)
        self.series = [torch.as_tensor(item, dtype=torch.float32).flatten() for item in series]
        if regression:
            self.labels = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32)
            self.label_length = 1 if self.labels.ndim == 1 else self.labels.shape[1]
        else:
            raw_labels = [str(label) for label in labels]
            unique_labels = sorted(set(raw_labels))
            self.label_to_index = {label: index for index, label in enumerate(unique_labels)}
            self.labels = torch.as_tensor([self.label_to_index[label] for label in raw_labels], dtype=torch.long)
            self.n_classes = len(unique_labels)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x = self.series[idx]
        if self.transform_to_patch:
            x = _to_patches(x, self.ratio_patches)
        y = self.labels[idx]
        return x, y


def _to_patches(series, ratio_patches):
    num_patches = max(1, int(ratio_patches))
    usable_length = (series.numel() // num_patches) * num_patches
    if usable_length == 0:
        padded = torch.zeros(num_patches)
        padded[: series.numel()] = series
        series = padded
        usable_length = num_patches
    return series[:usable_length].reshape(num_patches, usable_length // num_patches)


def _read_supervised_timeseries(path):
    if path.endswith(".ts"):
        return _read_ts_file(path)
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else df.columns[-1]
    features = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce").fillna(0)
    labels = df[label_col].to_numpy()
    return features.to_numpy(dtype=np.float32), labels


def _read_ts_file(path):
    series = []
    labels = []
    in_data = False
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if line.startswith("@") or not in_data:
                continue
            parts = line.split(":")
            labels.append(parts[-1].strip())
            values = []
            for channel in parts[:-1]:
                channel_values = [float(value) for value in channel.replace(",", " ").split() if value != "?"]
                values.extend(channel_values)
            series.append(values)
    return series, np.asarray(labels)


def get_eval_loaders(
    path_train,
    path_test,
    batch_size,
    ratio_patches=20,
    ratio_supervision=1.0,
    transform_to_patch=True,
    mask=False,
    mask_ratio=0.0,
    regression=False,
):
    """Build train/test DataLoaders for supervised downstream evaluation."""
    del mask, mask_ratio
    train_dataset = ClassificationDataset(path_train, ratio_patches, transform_to_patch, regression)
    test_dataset = ClassificationDataset(path_test, ratio_patches, transform_to_patch, regression)

    if 0 < ratio_supervision < 1:
        supervised_count = max(1, int(len(train_dataset) * ratio_supervision))
        train_dataset = Subset(train_dataset, range(supervised_count))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
