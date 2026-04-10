from typing import Iterator, Dict, Any, Optional, Callable, List
from torch.utils.data import DataLoader as TorchDataLoader
from .BaseDataLoader import BaseDataLoader
from .BaseDataset import BaseDataset
import torch
from torch.utils.data import Dataset as TorchDataset

# Imported lazily at call sites where needed to avoid requiring ML_Core on sys.path
# at import time. Users must have ML_Core/src/ on sys.path when building datasets.
from time_moe.datasets.time_moe_dataset import (
    TimeMoEMultiFrequencyWindowDatasetV2,
    MultiFrequencyTimeSeriesDataset,
)


def collate_multi_frequency_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that stacks tensors and keeps non-tensor fields as lists."""
    if not batch:
        return {}
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ('ticker', 'date_seq', 'prediction_sequence_keys', 'augmented_feature'):
            result[key] = [sample[key] for sample in batch]
        else:
            result[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return result


class TimeMoeTorchDataset(BaseDataset, TorchDataset):
    """Adapter that wraps TimeMoEMultiFrequencyWindowDatasetV2 as a PyTorch Dataset."""

    def __init__(self, window_dataset: TimeMoEMultiFrequencyWindowDatasetV2):
        self.window_dataset = window_dataset

    def __len__(self) -> int:
        return len(self.window_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.window_dataset[idx]


class TimeMoeDataLoader(BaseDataLoader):
    """DataLoader for multi-frequency time series."""

    def __init__(
        self,
        dataset: TimeMoeTorchDataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[list], Dict[str, Any]]] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self._dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn or collate_multi_frequency_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._dataloader)

    def __len__(self) -> int:
        return len(self._dataloader)


def create_dataloader(
    df,
    args,
    stage: str,
    max_length: int,
    stride: int,
    normalization_method: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    prediction_length: int = 0,
    resample_params: Optional[list] = None,
) -> TimeMoeDataLoader:
    """Build a TimeMoeDataLoader from a feature-engineered DataFrame.

    Args:
        df:                    Feature-engineered DataFrame (loaded from parquet).
        args:                  Config Namespace (from load_config / config_to_args).
        stage:                 One of 'train', 'test', 'inference'.
        max_length:            Context window length.
        stride:                Sliding window stride.
        normalization_method:  'zero' (z-score) or 'max'.
        batch_size:            Batch size for the DataLoader.
        shuffle:               Whether to shuffle the DataLoader.
        num_workers:           DataLoader worker processes.
        prediction_length:     Forecast horizon length.
        resample_params:       Optional resampling bucket configs for training.
    """
    base_dataset = MultiFrequencyTimeSeriesDataset(args, df, stage)
    window_dataset = TimeMoEMultiFrequencyWindowDatasetV2(
        base_dataset,
        context_length=max_length,
        prediction_length=prediction_length,
        stride=stride,
        normalization_method=normalization_method,
        inferencing_mode=(stage == 'inference'),
        resample_weights_path=resample_params or [],
    )
    torch_dataset = TimeMoeTorchDataset(window_dataset)
    return TimeMoeDataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
