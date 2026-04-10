#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import numpy as np
from argparse import Namespace
from .ts_dataset import TimeSeriesDataset
from .general_dataset import GeneralDataset
from .binary_dataset import BinaryDataset
import logging 
from ..models.configuration_time_moe import TimeMoeConfig
import pandas as pd
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import DataFrameIterator
import pickle
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import json
from ..base_classes.BaseDataLoader import BaseDataLoader, BaseDataset
from collections import defaultdict
from datetime import datetime

class TimeMoEDataset(TimeSeriesDataset):

    def __init__(self, data_folder, normalization_method=None):
        self.data_folder = data_folder
        self.normalization_method = normalization_method
        self.datasets = []
        self.num_tokens = None
        #logging.info(f'Data folder: {data_folder}')
        if normalization_method is None:
            self.normalization_method = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == 'max':
                self.normalization_method = max_scaler
            elif normalization_method.lower() == 'zero':
                self.normalization_method = zero_scaler
            else:
                raise ValueError(f'Unknown normalization method: {normalization_method}')
        else:
            self.normalization_method = normalization_method

        if BinaryDataset.is_valid_path(self.data_folder):
            #logging.info('Found as Binary Dataset')
            ds = BinaryDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        elif GeneralDataset.is_valid_path(self.data_folder):
            #logging.info('Found as General Dataset')
            ds = GeneralDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        else:
            logging.info('Walking Through')
            # walk through the data_folder
            for root, dirs, files in os.walk(self.data_folder):
                for file in files:
                    fn_path = os.path.join(root, file)
                    if file != BinaryDataset.meta_file_name and GeneralDataset.is_valid_path(fn_path):
                        ds = GeneralDataset(fn_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                for sub_folder in dirs:
                    folder_path = os.path.join(root, sub_folder)
                    if BinaryDataset.is_valid_path(folder_path):
                        ds = BinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)

        self.cumsum_lengths = [0]
        for ds in self.datasets:
            self.cumsum_lengths.append(
                self.cumsum_lengths[-1] + len(ds)
            )
        self.num_sequences = self.cumsum_lengths[-1]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
        elif seq_idx < 0:
            raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        seq = self.datasets[dataset_idx][dataset_offset]

        if self.normalization_method is not None:
            seq = self.normalization_method(seq)
        return seq

    def get_sequence_length_by_idx(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
        elif seq_idx < 0:
            raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_sequence_length_by_idx(dataset_offset)

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([ds.get_num_tokens() for ds in self.datasets])
        return self.num_tokens


###########################################################################################################################################
############################################ Mixed Frequency Time-MOE Dataset #############################################################
###########################################################################################################################################




##############################################################################
###################### Dataset Loader for Pre-Model Training #################
##############################################################################

class MultiFrequencyTimeSeriesDataset:
    '''
    Base dataset that holds multi-frequency time series data for each ticker
    '''
    def __init__(self, args: Namespace, feature_engineered_df: pd.DataFrame, stage: str, normalization_method=None):
        if stage == 'train':
            self.start_date = args.regression.S3.train_start_date
            self.end_date = args.regression.S3.train_end_date
        elif stage == 'test':
            self.start_date = args.regression.S3.test_start_date
            self.end_date = args.regression.S3.test_end_date
        elif stage == 'inference':
            self.start_date = args.regression.S3.inference_start_date
            self.end_date = args.regression.S3.inference_end_date
        elif stage == 'full':
            self.start_date = args.regression.S3.full_start_date
            self.end_date = args.regression.S3.full_end_date
            
        self.stage = stage
            
        # Channel configs: List[(patch_len, stride_len), ...]
        self.channel_configs = args.regression.S3.channel_configs
        self.channel_columns = args.regression.S3.channel_columns
        
        # Price channel is always first
        self.price_columns = self.channel_columns[0]
        self.price_patch_len, self.price_stride, _ = self.channel_configs[0]
            
        # Build sequences per ticker
        self.sequences = self._prepare_sequences(feature_engineered_df)
        
        
        
    def _prepare_sequences(self, feature_engineered_df: pd.DataFrame) -> List[pd.DataFrame]:
        '''
        Prepare full sequences for each ticker with all frequency channels
        '''
        tickers = feature_engineered_df['Ticker'].unique()
        all_sequences = []
        
        for ticker in tickers:
            ticker_channel = []
            ticker_df = feature_engineered_df[feature_engineered_df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
            
            # Get price channel data within date range
            price_df = ticker_df[self.price_columns + ['Date', 'Ticker']].copy()
            price_df = price_df[(price_df['Date'] >= self.start_date) & 
                                (price_df['Date'] <= self.end_date)]
            
            ticker_channel.append(price_df)
            for channel_idx in range(1, len(self.channel_columns)):
                channel_cols = self.channel_columns[channel_idx]
                patch_len, _ , patch_features = self.channel_configs[channel_idx]
                channel_dfs = ticker_df[channel_cols + ['Date', 'Ticker']].dropna().copy()
                ticker_channel.append(channel_dfs)
                
            all_sequences.append(ticker_channel)
        
        return all_sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def get_sequence_length_by_idx(self, idx):
        '''Get length of price sequence for a given ticker index'''
        return len(self.sequences[idx][0])
    
    

        
class TimeMoEMultiFrequencyWindowDatasetV2(BaseDataset):
    """
    A dataset class for generating sliding windows from multi-frequency time series data.
    Windows are computed on-the-fly in __getitem__ instead of being pre-computed.
    """

    def __init__(
        self, 
        dataset: MultiFrequencyTimeSeriesDataset, 
        context_length: int, 
        prediction_length: int = 0, 
        stride: int = 1,
        normalization_method='zero',
        inferencing_mode: bool = False,
        resample_weights_path: list = [],
    ):
        self.dataset = dataset
        self.context_length = dataset.channel_configs[0][0]
        self.prediction_length = prediction_length
        self.window_size = self.context_length
        self.window_size_plus_prediction = self.window_size + self.prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size
        
        # Store channel configs for tensor extraction
        self.channel_configs = dataset.channel_configs
        self.channel_columns = dataset.channel_columns
        
        if normalization_method is None:
            self.normalization_method = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == 'max':
                self.normalization_method = max_scaler
            elif normalization_method.lower() == 'zero':
                self.normalization_method = zero_scaler
            else:
                raise ValueError(f'Unknown normalization method: {normalization_method}')
        else:
            self.normalization_method = normalization_method
            
        self.inferencing_mode = inferencing_mode
        
        logging.info(f'MultiFrequencyTimeSeriesDatasetV2 initialized with Price context_length={self.context_length}, prediction_length={self.prediction_length}, window_size={self.window_size}, stride={self.stride}, inferencing_mode={self.inferencing_mode}')
        
        # Build index mapping: (ticker_idx, window_start_idx)
        
        # Check assertion all length of the price_df without nan is all same
        all_price_df_lens = [len(i[0].dropna()) for i in self.dataset]
        assert len(set(all_price_df_lens)) == 1, f'Not all ticker price lengths same: {all_price_df_lens}'
        
        self.window_indices = self._build_window_indices()
        if len(resample_weights_path) > 0:
            logging.info(f'Resampling Dataset according to {len(resample_weights_path)} rules')
            for param in resample_weights_path:
                # Resample the window_indices
                self.window_indices = self._resample_dataset(param)
            
        logging.info(f'Total windows available: {len(self.window_indices)}')
        
        
        # Date Index
        self._date_index_map = None
        self._date_mapping_built = False
        
    def _resample_dataset_blanket(self, weight: float):
        '''
        Blanket reasmpling of all the window indices
        '''
        num_samples = int(len(self.window_indices) * (weight))
        indices = np.arange(len(self.window_indices))
        sampled_indices = np.random.choice(indices, size=num_samples, replace=True)
        sampled_sequences = [self.window_indices[i] for i in sampled_indices]
        
        return sampled_sequences
        
        
        
    def _resample_dataset(self, resample_params: dict):
        '''
        List of weights that will let us resample by buckets
        '''
        logging.info(f'Resampling dataset, Current Dataset Length: {len(self.window_indices)}')        
        
        bucket_to_weights = resample_params.get('weights', {}).__dict__
        ticker_to_buckets_path = resample_params.get('buckets', {})
        
        if ticker_to_buckets_path == 'ALL':
            return self._resample_dataset_blanket(bucket_to_weights.get('0'))
        
        # Load the ticker to bucket mapping
        with open(ticker_to_buckets_path, 'r') as f:
            ticker_to_bucket = json.load(f)
        logging.info(f'Loaded Ticker to Bucket mapping from {ticker_to_buckets_path}')
        
        # Partition into their individual tickers first
        bucket_to_sequences = {}
        full_sequences = []
        for seq in self.window_indices:
            ticker = seq[2]
            bucket = ticker_to_bucket.get(ticker, None)
            if bucket is not None:
                if bucket not in bucket_to_sequences:
                    bucket_to_sequences[bucket] = []
                bucket_to_sequences[bucket].append(seq)
            else:
                logging.warning(f'Ticker {ticker} not found in bucket mapping, skipping resampling for this ticker.')
        
        
        for bucket, sequences in bucket_to_sequences.items():
            weight = bucket_to_weights.get(str(bucket), 1.0)
            if weight <= 0:
                logging.info(f'Bucket {bucket} has non-positive weight {weight}, skipping.')
                continue
            elif weight == 1.0:
                full_sequences.extend(sequences)
                #continue
            else:
                # Resample with replacement
                num_samples = int(len(sequences) * (weight))
                if num_samples > 0:
                    sampled_sequences = np.random.choice(sequences, size=num_samples, replace=True).tolist()
                    full_sequences.extend(sampled_sequences)
                    #logging.info(f'Resampled {num_samples} additional sequences for ticker {ticker} with weight {weight}.')
        
        logging.info(f'After resampling, New Dataset Length: {len(full_sequences)}')
        return full_sequences
    
    def _build_window_indices(self):
        """
        Build a list of (ticker_idx, window_start_idx) tuples for all valid windows.
        This gives us the total dataset length without pre-computing tensors.
        """
        window_indices = []
        
        for ticker_idx in range(len(self.dataset)):
            ticker_dfs = self.dataset[ticker_idx]
            price_df = ticker_dfs[0]
            n_points = len(price_df)
            tickers = price_df['Ticker'].unique()
            assert len(tickers) == 1, f'More than one ticker detected!'
            ticker = tickers[0]
            
            # Skip sequences with fewer than 2 points
            if n_points < 2:
                continue
            
            # Determine max index for windowing
            if self.inferencing_mode:
                max_idx = n_points - self.window_size_plus_one - self.prediction_length
            else:
                max_idx = n_points - self.window_size_plus_one + 1
            
            # Create window indices with stride
            for window_start in range(0, max_idx, self.stride):
                end_date = price_df.iloc[window_start + self.window_size_plus_one]['Date']
                window_indices.append((ticker_idx, window_start, ticker, end_date))
        
        return window_indices
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        """
        Compute window on-the-fly for the given index.
        Includes assertion checks for same length and no NaN values.
        """
        ticker_idx, window_start, ticker, _ = self.window_indices[idx]
        ticker_dfs = self.dataset[ticker_idx]
        
        # Extract price DataFrame (first in list)
        price_df = ticker_dfs[0]
        
        # Get price window
        price_window = price_df.iloc[window_start: window_start + self.window_size_plus_one]
        price_data_cols = [col for col in price_window.columns if col not in ['Date', 'Ticker']]
        price_seq = price_window[price_data_cols].values.astype(np.float32)
        date_seq = price_window['Date'].values
        end_date = date_seq[-1]
        
        # ASSERTION: Check for NaN in raw price data
        assert not np.isnan(price_seq).any(), \
            f'NaN detected in raw price data for ticker {ticker} at window_start {window_start}'
        
        # Normalize prices
        if self.normalization_method is not None:
            price_seq, mean_seq_price, std_seq_price = self.normalization_method(price_seq)
        
        # ASSERTION: Check for NaN after normalization
        assert not np.isnan(price_seq).any(), \
            f'NaN detected in normalized price data for ticker {ticker} at window_start {window_start}'
        
        # Create loss mask
        loss_mask = np.ones(len(price_seq) - 1, dtype=np.int32)
        
        # Handle padding if needed (shouldn't happen with assertions, but kept for safety)
        n_pad = self.window_size_plus_one - len(price_seq)
        if n_pad > 0:
            price_seq = np.pad(
                price_seq, 
                ((0, n_pad), (0, 0)), 
                'constant', 
                constant_values=0
            )
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)
        
        sequence_window = {
            'input_ids': torch.tensor(price_seq[:-1], dtype=torch.float32),
            'labels': torch.tensor(price_seq[1:], dtype=torch.float32),
            'loss_masks': torch.tensor(loss_mask, dtype=torch.float32),
            'ticker': ticker,
            'date_seq': date_seq
        }
        
        # ASSERTION: Check tensors for NaN
        #assert not torch.isnan(sequence_window['input_ids']).any(), f'NaN in input_ids for ticker {ticker} at window_start {window_start}'
        #assert not torch.isnan(sequence_window['labels']).any(), f'NaN in labels for ticker {ticker} at window_start {window_start}'
        
        # Process other channels
        for channel_idx in range(1, len(ticker_dfs)):
            channel_df = ticker_dfs[channel_idx]
            patch_len, _, patch_num_features = self.channel_configs[channel_idx]
            data_cols = [col for col in channel_df.columns if col not in ['Date', 'Ticker']]
            
            # Get data up to end date
            channel_df_sliced = channel_df[channel_df['Date'] <= end_date]
            
            if len(channel_df_sliced) < patch_len:
                # Pad if insufficient data
                channel_values = channel_df_sliced[data_cols].values.astype(np.float32)
                pad_len = patch_len - len(channel_values)
                channel_df_data = np.pad(
                    channel_values,
                    ((pad_len, 0), (0, 0)),
                    'constant',
                    constant_values=0.0
                )
                
                # Compute days_from_end
                if len(channel_df_sliced) > 0:
                    channel_dates = channel_df_sliced['Date'].values
                    days_from_end = (
                        pd.to_datetime(end_date) - pd.to_datetime(channel_dates)
                    ).days.astype(np.float32)
                    padded_dates = np.full(pad_len, 999999.0, dtype=np.float32)
                    channel_df_data_dates = np.concatenate([padded_dates, days_from_end])
                else:
                    channel_df_data_dates = np.full(patch_len, 999999.0, dtype=np.float32)
            else:
                # Take last patch_len values
                channel_df_data = channel_df_sliced[data_cols].iloc[-patch_len:].values.astype(np.float32)
                channel_df_data_dates = channel_df_sliced[['Date']].iloc[-patch_len:]
                channel_df_data_dates['days_from_end'] = (
                    (pd.to_datetime(end_date) - pd.to_datetime(channel_df_data_dates['Date'])).dt.days
                )
                channel_df_data_dates = channel_df_data_dates['days_from_end'].values.astype(np.float32)
            
            # ASSERTION: Check channel length
            #assert len(channel_df_data) == patch_len, f'Channel {channel_idx} length mismatch for ticker {ticker}: expected {patch_len}, got {len(channel_df_data)}'
            
            # ASSERTION: Check for NaN before normalization
            #assert not np.isnan(channel_df_data).any(), f'NaN in channel {channel_idx} raw data for ticker {ticker} at window_start {window_start}'
            
            # Apply normalization
            if self.normalization_method is not None:
                channel_df_data, _, _ = self.normalization_method(channel_df_data)
            
            # ASSERTION: Check for NaN after normalization
            #assert not np.isnan(channel_df_data).any(), f'NaN in channel {channel_idx} normalized data for ticker {ticker} at window_start {window_start}'
            
            sequence_window[f'channel_{channel_idx}'] = torch.tensor(channel_df_data, dtype=torch.float32)
            sequence_window[f'channel_{channel_idx}_dates'] = torch.tensor(channel_df_data_dates, dtype=torch.float32)
        
        # Handle inferencing mode
        if self.inferencing_mode:
            pred_end = window_start + self.window_size_plus_one + self.prediction_length
            if pred_end < len(price_df):
                pred_seq = price_df.iloc[window_start + self.window_size_plus_one: pred_end][price_data_cols].values.astype(np.float32)
                sequence_window['prediction_sequence'] = torch.tensor(pred_seq, dtype=torch.float32)
                sequence_window['prediction_sequence_keys'] = price_df.iloc[window_start + self.window_size_plus_one: pred_end][['Date', 'Ticker']].values
                if self.normalization_method is not None:
                    pred_seq, mean_seq_pred, std_seq_output = self.normalization_method(pred_seq)
                    sequence_window['mean'] = torch.tensor(mean_seq_pred, dtype=torch.float32)
                    sequence_window['std'] = torch.tensor(std_seq_output, dtype=torch.float32)
                    sequence_window['prediction_sequence_norm'] = torch.tensor(pred_seq, dtype=torch.float32)
        
        return sequence_window
    
    @staticmethod
    def inverse_scale(forecast: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, 
                     normalization_method: str = 'zero') -> torch.Tensor:
        """Inverse transform normalized forecasts back to original scale."""
        if normalization_method.lower() == 'zero':
            if std == 0:
                return forecast + mean
            else:
                return forecast * std + mean
        elif normalization_method.lower() == 'max':
            if mean == 0:
                return forecast
            else:
                return forecast * mean
        else:
            raise ValueError(f'Unknown normalization method: {normalization_method}')
        
        
    
    # Date Index Mapping for fast lookup
    def _build_date_index_mapping(self):
        """
        Build a mapping from end_date -> list of indices that have that end_date.
        This allows efficient lookup of all windows ending on a specific date.
        
        Returns:
            dict: {end_date: [idx1, idx2, ...]}
        """
        date_index_map = defaultdict(list)
        
        # Use augmented_indices if augmentation is enabled, otherwise use window_indices
        indices_to_iterate = self.window_indices
        
        for idx, (ticker_idx, window_start, ticker, *_) in enumerate(tqdm(
            indices_to_iterate, 
            desc="Building date index mapping",
            unit="window"
        )):
            ticker_dfs = self.dataset[ticker_idx]
            price_df = ticker_dfs[0]
            
            # Get the end date of this window
            window_end_idx = window_start + self.window_size
            if window_end_idx < len(price_df):
                end_date = price_df.iloc[window_end_idx]['Date']
                
                # Convert to string for consistent key format
                if isinstance(end_date, pd.Timestamp):
                    end_date_str = end_date.strftime('%Y-%m-%d')
                else:
                    end_date_str = str(end_date)
                    
                index_shifted = idx - 1  ## Add to make sure we dont get the same date
                if index_shifted >= 0: 
                    date_index_map[end_date_str].append(index_shifted)
        
        logging.info(f'Built date index mapping with {len(date_index_map)} unique dates')
        if len(date_index_map) > 0:
            logging.info(f'Date range: {min(date_index_map.keys())} to {max(date_index_map.keys())}')
        
        return dict(date_index_map)
    
    def _ensure_date_mapping_built(self):
        """
        Ensure the date mapping is built. Builds it lazily on first access.
        """
        if not self._date_mapping_built:
            logging.info('Building date index mapping (first access)...')
            self._date_index_map = self._build_date_index_mapping()
            self._date_mapping_built = True
        return self._date_index_map
    
    def get_available_dates(self):
        """
        Get list of all available end dates.
        
        Returns:
            list: Sorted list of available dates as strings
        """
        date_map = self._ensure_date_mapping_built()
        return sorted(date_map.keys())

    
    def getItemByDate(self, date, return_dict=False):
        """
        Get all tensors/windows that end on the specified date.
        """
        date_map = self._ensure_date_mapping_built()
        
        # Normalize date to string format
        if isinstance(date, (pd.Timestamp, datetime)):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Check if date exists
        if date_str not in date_map:
            available_dates = self.get_available_dates()
            if len(available_dates) > 0:
                logging.warning(
                    f"Date {date_str} not found in dataset. "
                    f"Available date range: {available_dates[0]} to {available_dates[-1]}. "
                    f"Total available dates: {len(available_dates)}"
                )
                return []
            else:
                logging.warning(f"Date {date_str} not found in dataset. No dates available.")
        
        # Get all indices for this date
        indices = date_map[date_str]
        
        logging.info(f'Retrieving {len(indices)} windows for date {date_str}')
        
        # Fetch all windows
        if return_dict:
            result = {}
            for idx in indices:
                tensor_dict = self.__getitem__(idx)
                ticker = tensor_dict['ticker']
                result[ticker] = tensor_dict
        else:
            result = []
            for idx in indices:
                tensor_dict = self.__getitem__(idx)
                result.append(tensor_dict)
        
        return result
    
    def getItemsByDateRange(self, start_date, end_date, return_dict=False):
        """
        Get all tensors/windows that end within the specified date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            return_dict: If True, return nested dict {date: {ticker: tensor}}
        
        Returns:
            If return_dict=True:
                dict: {date: {ticker: tensor_dict}}
            If return_dict=False:
                list: [tensor_dict1, tensor_dict2, ...]
        """
        date_map = self._ensure_date_mapping_built()
        
        # Normalize dates
        if isinstance(start_date, (pd.Timestamp, datetime)):
            start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            start_date_str = str(start_date)
        
        if isinstance(end_date, (pd.Timestamp, datetime)):
            end_date_str = end_date.strftime('%Y-%m-%d')
        else:
            end_date_str = str(end_date)
        
        # Get dates in range
        dates_in_range = [
            date for date in date_map.keys()
            if start_date_str <= date <= end_date_str
        ]
        
        logging.info(f'Found {len(dates_in_range)} dates in range {start_date_str} to {end_date_str}')
        
        if return_dict:
            result = {}
            for date in sorted(dates_in_range):
                result[date] = self.getItemByDate(date, return_dict=True)
        else:
            result = []
            for date in sorted(dates_in_range):
                result.extend(self.getItemByDate(date, return_dict=False))
        
        return result
    
    def getItemByTicker(self, ticker, return_dict=False):
        """
        Get all windows for a specific ticker.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            return_dict: If True, return dict keyed by date
        
        Returns:
            If return_dict=True:
                dict: {date: tensor_dict}
            If return_dict=False:
                list: [tensor_dict1, tensor_dict2, ...]
        """
        matching_indices = []
        
        for idx, (ticker_idx, window_start, window_ticker) in enumerate(self.window_indices):
            if window_ticker == ticker:
                matching_indices.append(idx)
        
        if not matching_indices:
            raise ValueError(f"Ticker {ticker} not found in dataset")
        
        logging.info(f'Found {len(matching_indices)} windows for ticker {ticker}')
        
        if return_dict:
            result = {}
            for idx in matching_indices:
                tensor_dict = self.__getitem__(idx)
                date = tensor_dict['date_seq'][-1]
                if isinstance(date, np.datetime64):
                    date = pd.Timestamp(date).strftime('%Y-%m-%d')
                result[str(date)] = tensor_dict
        else:
            result = [self.__getitem__(idx) for idx in matching_indices]
        
        return result
    
    def getItemByTickerAndDate(self, ticker, date):
        """
        Get the window for a specific ticker on a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date as string or datetime-like
        
        Returns:
            dict: Single tensor dictionary
        
        Raises:
            ValueError: If ticker/date combination not found
        """
        date_map = self._ensure_date_mapping_built()
        
        # Normalize date
        if isinstance(date, (pd.Timestamp, datetime)):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Get all windows for this date
        if date_str not in date_map:
            raise ValueError(f"Date {date_str} not found in dataset")
        
        indices = date_map[date_str]
        
        # Find matching ticker
        for idx in indices:
            _, _, window_ticker = self.window_indices[idx]
            if window_ticker == ticker:
                return self.__getitem__(idx)
        
        # Not found
        raise ValueError(f"No window found for ticker {ticker} on date {date_str}")
    
    def rebuild_date_mapping(self):
        """
        Manually rebuild the date index mapping.
        Useful after resampling or if window_indices has been modified.
        """
        logging.info('Manually rebuilding date index mapping...')
        self._date_index_map = self._build_date_index_mapping()
        self._date_mapping_built = True
        return self._date_index_map



class TimeMoEMultiFrequencyWindowDatasetV3(TimeMoEMultiFrequencyWindowDatasetV2):
    """
    Extended dataset class that generates synthetic training samples by mutating
    features based on quantiles from rolling window lookbacks.
    
    For each original window, generates additional samples with individual features
    changed to values from their historical distribution (quantiles).
    """

    def __init__(
        self, 
        dataset: MultiFrequencyTimeSeriesDataset, 
        context_length: int, 
        prediction_length: int = 0, 
        stride: int = 1,
        normalization_method='zero',
        inferencing_mode: bool = False,
        resample_weights_path: list = [],
        augmentation_config: dict = None,
    ):
        # Initialize parent class
        super().__init__(
            dataset=dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            normalization_method=normalization_method,
            inferencing_mode=inferencing_mode,
            resample_weights_path=resample_weights_path,
        )
        
        # Augmentation configuration
        self.augmentation_enabled = augmentation_config is not None and inferencing_mode
        self.augmentation_config = augmentation_config
        
        if self.augmentation_enabled:
            self.features_to_mutate = {feature: num_lookback for feature, num_lookback in augmentation_config.get('features_to_mutate')}
            self.num_quantile_points = self.augmentation_config.get('num_points', 10)
            
            self.augmentation_start_date = self.augmentation_config.get('start_date')
            self.augmentation_end_date = self.augmentation_config.get('end_date')
            
            # Build augmented index mapping
            self._build_augmented_indices()
            
            logging.info(f'Augmentation enabled with {len(self.features_to_mutate)} features, '
                        f'{self.num_quantile_points} quantile points')
            logging.info(f'Total samples after augmentation: {len(self.augmented_indices)}')
        else:
            self.augmented_indices = None
            logging.info('Augmentation disabled')
    
    def _build_augmented_indices(self):
        """
        Build augmented index mapping that includes:
        - Original windows (augmentation_idx = -1)
        - Synthetic windows for each feature mutation (augmentation_idx >= 0)
        
        Structure: List of tuples (ticker_idx, window_start, ticker, feature_name, quantile_idx)
        - feature_name = None for original samples
        - quantile_idx indicates which quantile value to use (0 to num_quantile_points-1)
        """
        logging.info(f'Building Augmented Indices...........')
        augmented_indices = []
        
        for ticker_idx, window_start, ticker, end_date in tqdm(self.window_indices, desc="Building augmented indices", unit="window"):
            
            # Skip if end_date not within date_range
            if not (pd.to_datetime(self.augmentation_start_date) <= end_date < pd.to_datetime(self.augmentation_end_date)):
                continue
            
            # Add original window
            augmented_indices.append((ticker_idx, window_start, ticker, None, -1))
            
            # Add synthetic samples for each feature
            for feature_name in self.features_to_mutate:
                # Check if feature exists in any channel for this ticker
                if not self._feature_exists_in_ticker(ticker_idx, feature_name):
                    continue
                
                # Generate samples for each quantile point
                for quantile_idx in range(self.num_quantile_points):
                    augmented_indices.append(
                        (ticker_idx, window_start, ticker, feature_name, quantile_idx)
                    )
        
        self.augmented_indices = augmented_indices
        logging.info(f'Built {len(augmented_indices)} total indices '
                    f'({len([x for x in augmented_indices if x[3] is None])} original, '
                    f'{len([x for x in augmented_indices if x[3] is not None])} augmented)')
    
    def _feature_exists_in_ticker(self, ticker_idx: int, feature_name: str) -> bool:
        """Check if a feature exists in any channel for the given ticker"""
        ticker_dfs = self.dataset[ticker_idx]
        
        # Check all channels except price (channel 0)
        for channel_idx in range(1, len(ticker_dfs)):
            channel_df = ticker_dfs[channel_idx]
            if feature_name in channel_df.columns:
                return True
        
        return False
    
    def _compute_quantile_values(
        self, 
        channel_df: pd.DataFrame, 
        feature_name: str, 
        end_date: np.datetime64
    ) -> np.ndarray:
        """
        Compute quantile values for a feature from rolling window lookback.
        
        Args:
            channel_df: DataFrame containing the feature
            feature_name: Name of the feature to compute quantiles for
            end_date: End date of the current window
            
        Returns:
            Array of quantile values (length = num_quantile_points)
        """
        # Filter data up to end_date and within lookback window
        end_date_pd = pd.to_datetime(end_date)
        start_date_lookback = end_date_pd - pd.Timedelta(days=self.features_to_mutate.get(feature_name, 1008))
        
        historical_data = channel_df[
            (channel_df['Date'] >= start_date_lookback) & 
            (channel_df['Date'] <= end_date_pd)
        ][feature_name].dropna()
        
        if len(historical_data) == 0:
            # If no historical data, return zeros
            return np.zeros(self.num_quantile_points, dtype=np.float32)
        
        # Compute quantiles
        quantile_positions = np.linspace(0, 1, self.num_quantile_points)
        quantile_values = np.quantile(historical_data.values, quantile_positions)
        
        return quantile_values.astype(np.float32)
    
    def _mutate_channel_data(
        self,
        channel_df_data: np.ndarray,
        channel_df: pd.DataFrame,
        feature_name: str,
        quantile_idx: int,
        end_date: np.datetime64,
        data_cols: list
    ) -> np.ndarray:
        """
        Mutate a specific feature in channel data with a quantile value.
        
        Args:
            channel_df_data: Current channel data array
            channel_df: Original channel DataFrame
            feature_name: Feature to mutate
            quantile_idx: Which quantile to use
            end_date: End date for lookback window
            data_cols: List of data column names
            
        Returns:
            Mutated channel data array
        """
        if feature_name not in data_cols:
            return channel_df_data
        
        # Compute quantile values
        quantile_values = self._compute_quantile_values(channel_df, feature_name, end_date)
        
        # Get the specific quantile value to use
        mutation_value = quantile_values[quantile_idx]
        
        # Find feature index in data_cols
        feature_idx = data_cols.index(feature_name)
        
        # Create mutated copy
        mutated_data = channel_df_data.copy()
        
        # Replace all values of this feature with the quantile value
        # This affects all timesteps in the patch
        mutated_data[:, feature_idx] = mutation_value
        
        return mutated_data
    
    def __len__(self):
        if self.augmentation_enabled:
            return len(self.augmented_indices)
        else:
            return len(self.window_indices)
    
    def __getitem__(self, idx):
        """
        Compute window on-the-fly with optional feature mutation.
        """
        if not self.augmentation_enabled:
            # Use parent implementation if augmentation disabled
            return super().__getitem__(idx)
        
        # Unpack augmented index
        ticker_idx, window_start, ticker, feature_to_mutate, quantile_idx = self.augmented_indices[idx]
        
        ticker_dfs = self.dataset[ticker_idx]
        
        # Extract price DataFrame (first in list)
        price_df = ticker_dfs[0]
        
        # Get price window
        price_window = price_df.iloc[window_start: window_start + self.window_size_plus_one]
        price_data_cols = [col for col in price_window.columns if col not in ['Date', 'Ticker']]
        price_seq = price_window[price_data_cols].values.astype(np.float32)
        date_seq = price_window['Date'].values
        end_date = date_seq[-1]
        
        # ASSERTION: Check for NaN in raw price data
        assert not np.isnan(price_seq).any(), \
            f'NaN detected in raw price data for ticker {ticker} at window_start {window_start}'
        
        # Normalize prices
        if self.normalization_method is not None:
            price_seq, mean_seq_price, std_seq_price = self.normalization_method(price_seq)
        
        # ASSERTION: Check for NaN after normalization
        assert not np.isnan(price_seq).any(), \
            f'NaN detected in normalized price data for ticker {ticker} at window_start {window_start}'
        
        # Create loss mask
        loss_mask = np.ones(len(price_seq) - 1, dtype=np.int32)
        
        # Handle padding if needed
        n_pad = self.window_size_plus_one - len(price_seq)
        if n_pad > 0:
            price_seq = np.pad(
                price_seq, 
                ((0, n_pad), (0, 0)), 
                'constant', 
                constant_values=0
            )
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)
        
        sequence_window = {
            'input_ids': torch.tensor(price_seq[:-1], dtype=torch.float32),
            'labels': torch.tensor(price_seq[1:], dtype=torch.float32),
            'loss_masks': torch.tensor(loss_mask, dtype=torch.float32),
            'ticker': ticker,
            'date_seq': date_seq,
            #'is_augmented': feature_to_mutate is not None,
            'augmented_feature': feature_to_mutate if feature_to_mutate else 'none',
        }
        
        # Process other channels with optional mutation
        for channel_idx in range(1, len(ticker_dfs)):
            channel_df = ticker_dfs[channel_idx]
            patch_len, _, patch_num_features = self.channel_configs[channel_idx]
            data_cols = [col for col in channel_df.columns if col not in ['Date', 'Ticker']]
            
            # Get data up to end date
            channel_df_sliced = channel_df[channel_df['Date'] <= end_date]
            
            if len(channel_df_sliced) < patch_len:
                # Pad if insufficient data
                channel_values = channel_df_sliced[data_cols].values.astype(np.float32)
                pad_len = patch_len - len(channel_values)
                channel_df_data = np.pad(
                    channel_values,
                    ((pad_len, 0), (0, 0)),
                    'constant',
                    constant_values=0.0
                )
                
                # Compute days_from_end
                if len(channel_df_sliced) > 0:
                    channel_dates = channel_df_sliced['Date'].values
                    days_from_end = (
                        pd.to_datetime(end_date) - pd.to_datetime(channel_dates)
                    ).days.astype(np.float32)
                    padded_dates = np.full(pad_len, 999999.0, dtype=np.float32)
                    channel_df_data_dates = np.concatenate([padded_dates, days_from_end])
                else:
                    channel_df_data_dates = np.full(patch_len, 999999.0, dtype=np.float32)
            else:
                # Take last patch_len values
                channel_df_data = channel_df_sliced[data_cols].iloc[-patch_len:].values.astype(np.float32)
                channel_df_data_dates = channel_df_sliced[['Date']].iloc[-patch_len:]
                channel_df_data_dates['days_from_end'] = (
                    (pd.to_datetime(end_date) - pd.to_datetime(channel_df_data_dates['Date'])).dt.days
                )
                channel_df_data_dates = channel_df_data_dates['days_from_end'].values.astype(np.float32)
            
            # Apply feature mutation if this is an augmented sample
            if feature_to_mutate is not None and quantile_idx >= 0:
                channel_df_data = self._mutate_channel_data(
                    channel_df_data=channel_df_data,
                    channel_df=channel_df_sliced,
                    feature_name=feature_to_mutate,
                    quantile_idx=quantile_idx,
                    end_date=end_date,
                    data_cols=data_cols
                )
            
            # Apply normalization
            if self.normalization_method is not None:
                channel_df_data, _, _ = self.normalization_method(channel_df_data)
            
            sequence_window[f'channel_{channel_idx}'] = torch.tensor(channel_df_data, dtype=torch.float32)
            sequence_window[f'channel_{channel_idx}_dates'] = torch.tensor(channel_df_data_dates, dtype=torch.float32)
        
        # Handle inferencing mode (same as parent)
        if self.inferencing_mode:
            pred_end = window_start + self.window_size_plus_one + self.prediction_length
            if pred_end < len(price_df):
                pred_seq = price_df.iloc[window_start + self.window_size_plus_one: pred_end][price_data_cols].values.astype(np.float32)
                sequence_window['prediction_sequence'] = torch.tensor(pred_seq, dtype=torch.float32)
                sequence_window['prediction_sequence_keys'] = price_df.iloc[window_start + self.window_size_plus_one: pred_end][['Date', 'Ticker']].values
                if self.normalization_method is not None:
                    pred_seq, mean_seq_pred, std_seq_output = self.normalization_method(pred_seq)
                    sequence_window['mean'] = torch.tensor(mean_seq_pred, dtype=torch.float32)
                    sequence_window['std'] = torch.tensor(std_seq_output, dtype=torch.float32)
                    sequence_window['prediction_sequence_norm'] = torch.tensor(pred_seq, dtype=torch.float32)
        
        return sequence_window

class MultiFrequencyDatasetLoader(BaseDataLoader):
    '''
    DataLoader wrapper for multi-frequency time series with TimeMoE windowing
    '''
    def __init__(
        self, 
        args: Namespace, 
        feature_engineered_df: pd.DataFrame, 
        stage: str,
        context_length: int,
        prediction_length: int = 0,
        stride: int = 1,
        batch_size: int = 32, 
        shuffle: bool = True,
        num_workers: int = 0
    ):
        # Create base dataset
        base_dataset = MultiFrequencyTimeSeriesDataset(args, feature_engineered_df, stage)
        
        # Create windowed dataset
        
        self.dataset = TimeMoEMultiFrequencyWindowDatasetV2(
                base_dataset, 
                context_length, 
                prediction_length,
                stride,
                inferencing_mode=True if stage in ['inference', 'full'] else False,
        )
            
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_multi_frequency_fn
        )
        logging.info(f'Created DataLoader with {len(self.dataset)} samples, batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}')
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    @staticmethod
    def collate_multi_frequency_fn(batch):
        '''
        Custom collate function for batching multi-frequency windows
        '''
        keys = batch[0].keys()
        result = {}
        
        for key in keys:
            if key in ['ticker', 'start_date', 'end_date', 'date_seq', 'prediction_sequence_keys', 'augmented_feature']:
                result[key] = [sample[key] for sample in batch]
            else:
                # Stack numerical arrays
                # print(f"{key}: {[sample[key].shape for sample in batch]}")  # Commented out for performance
                result[key] = torch.stack([sample[key] for sample in batch], dim=0)
        
        #print(f'Collated Batch Keys: {list(result.keys())}')
        return result
    


class MultiFrequencyDatasetLoaderV2():
    '''
    DataLoader wrapper for multi-frequency time series with TimeMoE windowing
    '''
    def __init__(
        self, 
        args: Namespace, 
        feature_engineered_df: pd.DataFrame, 
        stage: str,
        context_length: int,
        prediction_length: int = 0,
        stride: int = 1,
        batch_size: int = 32, 
        shuffle: bool = True,
        num_workers: int = 0
    ):
        # Create base dataset
        base_dataset = MultiFrequencyTimeSeriesDataset(args, feature_engineered_df, stage)
        
        
        # Get Augmentation Config
        augmentation_config = args.regression.S6.__dict__
        
        # Create windowed dataset
        self.dataset = TimeMoEMultiFrequencyWindowDatasetV3(
                base_dataset, 
                context_length, 
                prediction_length,
                stride,
                inferencing_mode=True if stage in ['inference', 'full'] else False,
                augmentation_config=augmentation_config
        )
            
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=MultiFrequencyDatasetLoaderV2.collate_multi_frequency_fn
        )
        logging.info(f'Created DataLoader with {len(self.dataset)} samples, batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}')
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    @staticmethod
    def collate_multi_frequency_fn(batch):
        '''
        Custom collate function for batching multi-frequency windows
        '''
        keys = batch[0].keys()
        result = {}
        
        for key in keys:
            if key in ['ticker', 'start_date', 'end_date', 'date_seq', 'prediction_sequence_keys', 'augmented_feature']:
                result[key] = [sample[key] for sample in batch]
            else:
                # Stack numerical arrays
                # print(f"{key}: {[sample[key].shape for sample in batch]}")  # Commented out for performance
                result[key] = torch.stack([sample[key] for sample in batch], dim=0)
        
        #print(f'Collated Batch Keys: {list(result.keys())}')
        return result
    
#######################################################################
###################### Dataset Loader for Inferencing #################
#######################################################################



class SlidingWindowDataset(Dataset):
    
    def __init__(self, args: Namespace, inference_sequences_path: str, inference_keys_path: str, batch_size: int, 
                        context_length: int, prediction_length: int):
        
        with open(inference_sequences_path, "rb") as f:
            self.inference_sequences = pickle.load(f)
        with open(inference_keys_path, "rb") as f:
            self.inference_keys = pickle.load(f)
        
        self.seqs = [entry["sequence"] for entry in self.inference_sequences]
        self.keys = [entry["keys"] for entry in self.inference_keys]

        self.samples = []
        
        for seq, key in zip(self.seqs, self.keys):
            seq = torch.tensor(seq, dtype=torch.float32)
            assert not torch.isnan(seq).any()

            for i in range(0, len(seq) - context_length - prediction_length, 1):
                sub = seq[i:i + context_length]                           # [context_len, num_features]
                pred_seq = seq[i + context_length: i + context_length + prediction_length]
                sub_keys = key[i:i + context_length]
                pred_seq_keys = key[i + context_length: i + context_length + prediction_length]

                # --- normalize per subsequence ---
                mean_output = sub[:, 0].mean().unsqueeze(0).unsqueeze(0)   # [1, 1]
                std_output = sub[:, 0].std().unsqueeze(0).unsqueeze(0)     # [1, 1]
                if std_output.item() == 0:
                    logging.warning(f'Warning! Error in creating Mean and Std')
                    std_output = torch.ones_like(std_output)
                    
                # Mean and STD for normalizing all the features
                mean = sub.mean(dim=0, keepdim=True)  # [1, F]
                std = sub.std(dim=0, keepdim=True)    # [1, F]
                
                # Replace any zeros in std with 1 (to avoid division by zero)
                std_safe = std.clone()
                std_safe[std_safe == 0] = 1.0

                sub_norm = (sub - mean) / std_safe
                pred_seq_norm = (pred_seq - mean) / std_safe
                
                # --- assertions to catch NaNs ---
                assert not torch.isnan(sub).any(), "NaN detected in raw subsequence!"
                assert not torch.isnan(pred_seq).any(), "NaN detected in raw prediction sequence!"
                assert not torch.isnan(sub_norm).any(), "NaN detected in normalized subsequence!"
                assert not torch.isnan(pred_seq_norm).any(), "NaN detected in normalized prediction sequence!"
                assert not torch.isnan(mean).any(), "NaN detected in mean computation!"
                assert not torch.isnan(std).any(), "NaN detected in std computation!"
                assert not torch.isnan(mean_output).any(), "NaN detected in mean_output!"
                assert not torch.isnan(std_output).any(), "NaN detected in std_output!"

                self.samples.append({
                    "subsequence_norm": sub_norm,                          # normalized [context_len, F]
                    "subsequence": sub,  
                    "subsequence_keys": sub_keys,                     # [context_len, num_keys]
                    "prediction_sequence_norm": pred_seq_norm,             # normalized [prediction_len, F]
                    "prediction_sequence": pred_seq,
                    "prediction_sequence_keys": pred_seq_keys,        # [prediction_len, num_keys]
                    "mean": mean_output,                                     # store per-subsequence mean []
                    "std": std_output                                        # store per-subsequence std []
                })

        logging.info(f"Created {len(self.samples)} sliding window sequences for inference.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

    def inverse_scale(self, forecast: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """
        Inverse Scaling for forecast: [prediction_len, F] or [B, prediction_len, F]
        forecast : normalized predictions
        mean, std : per-subsequence values stored in sample
        """
        return forecast * std + mean
        
        
        
def custom_collate_fn(batch):
    """
    Custom collate function that stacks numeric tensors 
    but leaves timestamp keys as Python objects.
    """
    subsequences = torch.stack([item["subsequence"] for item in batch])   # [B, context_len, features]
    prediction_sequences = torch.stack([item["prediction_sequence"] for item in batch]) # [B, pred_len, features]
    
    subsequences_norm = torch.stack([item["subsequence_norm"] for item in batch])   # [B, context_len, features]
    prediction_sequences_norm = torch.stack([item["prediction_sequence_norm"] for item in batch]) # [B, pred_len, features]

    subsequence_keys = [item["subsequence_keys"] for item in batch]       # list of timestamps
    prediction_sequence_keys = [item["prediction_sequence_keys"] for item in batch]
    subsequence_means = [item["mean"] for item in batch]
    subsequence_stds = [item["std"] for item in batch]

    return {
        "subsequence": subsequences,
        "prediction_sequence": prediction_sequences,
        "subsequence_norm": subsequences_norm,
        "prediction_sequence_norm": prediction_sequences_norm,
        "subsequence_keys": subsequence_keys,
        "prediction_sequence_keys": prediction_sequence_keys,
        "subsequence_means": subsequence_means,
        "subsequence_stds": subsequence_stds
    }
    
        
class InferenceTensorLoader:
    def __init__(self, args: Namespace, inference_sequences_path: str, inference_keys_path: str, 
                 batch_size: int, context_length: int, prediction_length: int):

        # Build dataset
        self.dataset = SlidingWindowDataset(args, inference_sequences_path, inference_keys_path, batch_size, 
                                            context_length, prediction_length)

        # Build DataLoader
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=custom_collate_fn)

    def get_loader(self):
        return self.loader
        
    
    
def zero_scaler(seq):
    """Standardize sequence (z-score normalization). Returns scaled seq, mean, std."""
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype

    mean_val = seq.mean()
    std_val = seq.std()
    
    # Add epsilon to prevent division by zero
    epsilon = 1e-8
    
    if std_val < epsilon:
        normed_seq = seq - mean_val
    else:
        normed_seq = (seq - mean_val) / (std_val + epsilon)
    
    # Clip to prevent extreme values
    normed_seq = np.clip(normed_seq, -10, 10)

    return normed_seq.astype(origin_dtype), mean_val, std_val


def zero_inverse_scaler(normed_seq, mean_val, std_val):
    """Inverse of zero_scaler."""
    if not isinstance(normed_seq, np.ndarray):
        normed_seq = np.array(normed_seq)
    
    epsilon = 1e-8
    if std_val < epsilon:
        original_seq = normed_seq + mean_val
    else:
        original_seq = normed_seq * std_val + mean_val

    return original_seq


def max_scaler(seq):
    """Scale sequence by its max absolute value. Returns scaled seq, max."""
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype

    max_val = np.abs(seq).max()
    epsilon = 1e-8
    
    if max_val < epsilon:
        normed_seq = seq
    else:
        normed_seq = seq / (max_val + epsilon)
    
    # Clip to prevent extreme values
    normed_seq = np.clip(normed_seq, -1, 1)

    return normed_seq.astype(origin_dtype), max_val, None  # Return None for std to match interface


def max_inverse_scaler(normed_seq, max_val):
    """Inverse of max_scaler."""
    if not isinstance(normed_seq, np.ndarray):
        normed_seq = np.array(normed_seq)
    
    if max_val == 0:
        original_seq = normed_seq
    else:
        original_seq = normed_seq * max_val

    return original_seq

def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    best_index = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= value:
            best_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_index
