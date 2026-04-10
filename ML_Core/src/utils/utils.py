import argparse
import yaml
import re
import pandas as pd
from typing import Iterator
from argparse import Namespace  
from torch.utils.data import Dataset, DataLoader
import torch

##########################################################################################################################
################################################# Config Arguments #######################################################
##########################################################################################################################


def load_config(path: str) -> dict:
    """Load YAML config into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def replace_placeholders(cfg: dict, placeholders: dict) -> dict:
    """Recursively replace placeholders in cfg using placeholders dict."""
    def _replace(value):
        if isinstance(value, str):
            for k, v in placeholders.items():
                value = value.replace(f"{{{k}}}", str(v))
            return value
        elif isinstance(value, dict):
            return {k: _replace(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_replace(v) for v in value]
        else:
            return value
    return _replace(cfg)


def config_to_args(cfg: dict, placeholders: dict = None) -> argparse.Namespace:
    """Convert dict to argparse.Namespace with optional placeholder replacement."""
    if placeholders:
        cfg = replace_placeholders(cfg, placeholders)

    def _dict_to_namespace(d):
        if isinstance(d, dict):
            return argparse.Namespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [_dict_to_namespace(v) for v in d]
        else:
            return d

    return _dict_to_namespace(cfg)


##########################################################################################################################
################################################# Torch Utilities #########################################################
##########################################################################################################################

def patch_torch_load():
    """Monkey-patch torch.load to use weights_only=False for all loads (required for older checkpoints)."""
    import torch
    _original = torch.load

    def _patched(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return _original(f, *args, **kwargs)

    torch.load = _patched


##########################################################################################################################
################################################# DataFrame Iterator #####################################################
##########################################################################################################################

class DataFrameIterator:
    def __init__(self, args: Namespace, df: pd.DataFrame, stage: str = 'train'):
        """
        df: input dataframe with 'Date' and 'Ticker' columns
        n_dates: number of unique dates per batch
        n_tickers: number of unique tickers per batch
        """
        self.df = df.copy()
        self.n_dates = args.regression.S3.window_size
        self.n_tickers = args.regression.S3.ticker_size
        
        if stage == 'train':
            start_date, end_date = args.regression.S3.train_start_date, args.regression.S3.train_end_date
        elif stage == 'test':
            start_date, end_date = args.regression.S3.test_start_date, args.regression.S3.test_end_date
        elif stage == 'inference':
            start_date, end_date = args.regression.S3.inference_start_date, args.regression.S3.inference_end_date

        # sort by date and ticker to maintain order
        self.df = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)]
        self.df = self.df.sort_values(by=['Date', 'Ticker'])
        
        self.unique_dates = self.df['Date'].drop_duplicates().tolist()
        self.unique_tickers = self.df['Ticker'].drop_duplicates().tolist()

        self.date_idx = 0
        self.ticker_idx = 0
        
        
    def __iter__(self) -> Iterator[pd.DataFrame]:
        return self

    def __next__(self) -> pd.DataFrame:
        if self.date_idx >= len(self.unique_dates):
            raise StopIteration

        # select date slice
        end_date_idx = min(self.date_idx + self.n_dates, len(self.unique_dates))
        date_slice = self.unique_dates[self.date_idx:end_date_idx]

        # select ticker slice
        end_ticker_idx = min(self.ticker_idx + self.n_tickers, len(self.unique_tickers))
        ticker_slice = self.unique_tickers[self.ticker_idx:end_ticker_idx]

        # get batch
        batch = self.df[
            self.df['Date'].isin(date_slice) &
            self.df['Ticker'].isin(ticker_slice)
        ].copy()

        # advance ticker index
        self.ticker_idx += self.n_tickers
        if self.ticker_idx >= len(self.unique_tickers):
            self.ticker_idx = 0
            self.date_idx += 1

        return batch
    


class TensorKeyDataset(Dataset):
    def __init__(self, all_tensors):
        
        
        # all_tensors is a list of (X, y, unique_keys)
        self.X_all = torch.cat([x for x, _, _ in all_tensors], dim=0)
        self.y_all = torch.cat([y for _, y, _ in all_tensors], dim=0)
        self.keys_all = sum([keys for _, _, keys in all_tensors], [])
        
        assert len(self.X_all) == len(self.y_all) == len(self.keys_all), \
            f"Mismatch between X, y, and keys length X: {len(self.X_all)}, Y: {len(self.y_all)}, Keys_all: {len(self.keys_all)}"
        
    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        return self.X_all[idx], self.y_all[idx], self.keys_all[idx]

