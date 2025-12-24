# Generic imports
import numpy as np
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Any

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler

# PyTorch Lightning
from lightning.pytorch import LightningDataModule

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from lightning.pytorch import LightningDataModule


def generate_time_features(timestamps):
    timestamps = pd.to_datetime(timestamps)
    time_feats = pd.DataFrame()
    time_feats["month"] = timestamps.dt.month / 12.0
    time_feats["quarter"] = timestamps.dt.quarter / 4.0
    time_feats["year_sin"] = np.sin(2 * np.pi * timestamps.dt.dayofyear / 365.25)
    time_feats["year_cos"] = np.cos(2 * np.pi * timestamps.dt.dayofyear / 365.25)
    return time_feats.astype(np.float32).values


class WindowForecastingDataset(Dataset):
    def __init__(
        self, series, time_feats=None, context_len=96, horizon=48, min_valid=12
    ):
        self.series = np.asarray(series).astype(np.float32)
        self.time_feats = (
            np.asarray(time_feats).astype(np.float32)
            if time_feats is not None
            else None
        )
        self.context_len = context_len
        self.horizon = horizon
        self.min_valid = min_valid
        self.total_len = len(series)

        self.num_samples = self.total_len - self.context_len - self.horizon + 1
        print(
            f"Number of samples: {self.num_samples}, Series length: {self.series.shape[0]}"
        )
        if self.num_samples <= 0:
            raise ValueError(
                "Time series too short for the given context and horizon lengths."
            )

    def __len__(self):
        # Set any fixed number, but not too large
        return 10000

    def __getitem__(self, idx):
        # Fixing random seed is important in order
        # not to get nan values
        np.random.seed(idx)
        # Vary valid length between min_valid and context_len
        valid_len = np.random.randint(self.min_valid, self.context_len + 1)
        pad_len = self.context_len - valid_len

        x = self.series[:valid_len]
        y = self.series[valid_len : valid_len + self.horizon]

        if self.time_feats is not None:
            x_time = self.time_feats[:valid_len]
        else:
            x_time = None

        x = torch.from_numpy(x).unsqueeze(-1)  # [valid_len, 1]
        y = torch.from_numpy(y)  # [horizon]

        if x_time is not None:
            x_time = torch.from_numpy(x_time)  # [valid_len, T]
            x = torch.cat([x, x_time], dim=-1)  # [valid_len, 1+T]

        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, x.shape[1])
            x = torch.cat([pad_tensor, x], dim=0)  # [context_len, F]

        pad_mask = torch.zeros(self.context_len, dtype=torch.bool)
        if pad_len > 0:
            pad_mask[:pad_len] = 1  # valid tokens are at the end

        return x, y, pad_mask


class WindowForecastingDataModule(LightningDataModule):
    def __init__(
        self,
        series: str,  # CSV path
        context_len: int = 96,
        horizon: int = 48,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.csv_path = series
        self.context_len = context_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.save_hyperparameters()

    def prepare_data(self):
        data_ = self.csv_path.replace(".csv", "")
        df = pd.read_csv(self.csv_path)

        df = df.rename(columns={"Date (UTC)": "ds", "Flux": "y"})
        df["unique_id"] = data_
        df["ds"] = pd.to_datetime(df["ds"], format="%m/%d/%Y")
        df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
        df["y"] = df["y"] / df["y"].max()

        time_feats = generate_time_features(df["ds"])

        self.series_raw = (
            df["y"]
            .iloc[: -self.horizon]
            .reset_index(drop=True)
            .values.astype(np.float32)
        )
        self.time_feats = time_feats[: -self.horizon]

        self.holdout_y = (
            df["y"]
            .iloc[-self.horizon :]
            .reset_index(drop=True)
            .values.astype(np.float32)
        )

    def setup(self, stage=None):
        if not hasattr(self, "series_raw"):
            self.prepare_data()

        self.dataset = WindowForecastingDataset(
            series=self.series_raw,
            time_feats=self.time_feats,
            context_len=self.context_len,
            horizon=self.horizon,
        )

        total_len = len(self.dataset)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len

        self.train_dataset = Subset(self.dataset, range(train_len))
        self.val_dataset = Subset(self.dataset, range(train_len, total_len))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def get_holdout(self):
        return self.holdout_y

