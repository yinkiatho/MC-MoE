import pandas as pd
import numpy as np
import json
import datetime
from argparse import Namespace
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.utils import DataFrameIterator
import torch
from models.mlp_dummy import MLPModel
import os
import pickle
from models.model_trainer import ModelTrainer
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from typing import List, Tuple, Dict, Optional
from time_moe.datasets.time_moe_dataset import *



#################################################################################################################################################################
########################################################### Time MOE ############################################################################################
#################################################################################################################################################################

def S3_tensor_stages_preparation(args: Namespace, feature_engineered_df: pd.DataFrame):
    
    logging.info(f"Tensor Stages Preparation for Time-MOE......... ")
    
    # Do any further preprocessing here
    tensor_stages = {'train': None, 'test': None, 'inference': None}
    for stage in tensor_stages:
        full_tensors = batched_data_preparation_time_moe(args, feature_engineered_df, stage=stage)
        tensor_stages[stage] = full_tensors
        
    if args.regression.S3.save_sequences: 
        logging.info(f'Saving the individual prepped sequences')
        # We save individually as {version_name}_{stage}_sequences and {version_name}_{stage}_keys.pkl
        
        for stage, sequences in tensor_stages.items():
            sequence = [{'sequence': sequence['sequence']} for sequence in sequences]
            keys = [{'keys': sequence['keys']} for sequence in sequences]
            seq_path = f"../data/processed_data/{args.regression.common.version_name}_{stage}_sequences.pkl"
            keys_path = f"../data/processed_data/{args.regression.common.version_name}_{stage}_keys.pkl"
            
            # Save to pickle
            with open(seq_path, "wb") as f:
                pickle.dump(sequence, f)
            with open(keys_path, "wb") as f:
                pickle.dump(keys, f)
        
    return tensor_stages 
        


def batched_data_preparation_time_moe(args: Namespace, feature_engineered_df: pd.DataFrame, stage='train'):
    '''
    Further preparation of training data such as upsampling etc and then prepping to tensors, we do by each ticker one unique sequence
    '''
    logging.info(f"Preparing Batched Sequences for Input for Time-MOE, current DataFrame Rows: {len(feature_engineered_df)}, for Stage {stage}")
    
    if stage == 'train':
        start_date, end_date = args.regression.S3.train_start_date, args.regression.S3.train_end_date
    elif stage == 'test':
        start_date, end_date = args.regression.S3.test_start_date, args.regression.S3.test_end_date
    elif stage == 'inference':
        start_date, end_date = args.regression.S3.inference_start_date, args.regression.S3.inference_end_date
        
    sliced_df = feature_engineered_df[(feature_engineered_df['Date'] >= start_date) & (feature_engineered_df['Date'] < end_date)].copy()
    
    all_sequences = []
    for ticker in sliced_df['Ticker'].unique():
        ticker_df = sliced_df[sliced_df['Ticker'] == ticker].copy()
        
        # Get the keys and values set
        unique_keys = ticker_df[['Date', 'Ticker']].copy()
        values = ticker_df.drop(columns=['Date', 'Ticker']).astype(float)
        
        # Rearrange the values df
        values = values[[args.regression.S2.original_label] + [i for i in values.columns if i != args.regression.S2.original_label]]
        
        unique_keys, values = unique_keys.values.tolist(), values.values.tolist()
        all_sequences.append({'sequence': values, 'keys': unique_keys})
    
    logging.info(f"Prepared Patched Tensors, Total Number of Patches: {len(all_sequences)}")
    return all_sequences 
        


def resample_batched_inputs(args: Namespace, all_inputs: list):
    '''
    Function to resample the batched inputs of X, y dataframes
    '''
    return all_inputs


 
def S3_model_inferencing_multi_frequency(args: Namespace, feature_engineered_df: pd.DataFrame = None, model = None, device=None):
    '''
    Model inferencing code
    '''
    logging.info(f"Inititating Model Inferencing....................")
    if model is None:
        logging.info(f"Model set as None! Loading from HuggingFace @ {args.regression.Time_MOE.model_path}")
        model = TimeMoeForPrediction.from_pretrained(
            args.regression.Time_MOE.model_path, 
            device_map='cpu' if not torch.cuda.is_available() else ('auto' if device is None else device),
            torch_dtype='auto',
        )
        
    if feature_engineered_df is None:
        feature_engineered_df = pd.read_parquet(args.regression.S2.feature_engineered_file_path)
        
    # Preparing the inferencing sequences via the DataLoader
    inference_loader = MultiFrequencyDatasetLoader(
        args=args,
        feature_engineered_df=feature_engineered_df,
        stage='inference',
        batch_size=args.regression.Time_MOE.inference_model_args.batch_size,
        context_length=args.regression.Time_MOE.train_model_args.max_length,
        prediction_length=args.regression.Time_MOE.inference_model_args.prediction_length,
    )
    all_sequences = []
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            #print(f'Input Tensor Shape: {batch['input_ids'].shape}')
            #print(f'Input IDs: {batch['input_ids']}')
            
            # Check for other channels
            # for key in batch:
            #     if key.startswith('channel_'):
            #         #print(f'Channel {key} Shape: {batch[key].shape}')
        
            # Check for NaNs
            if torch.isnan(batch['input_ids']).any():
                print("⚠️ NaN detected in input subsequence_norm!")
                
            # Enter the channels as kwargs
            output = model(
                input_ids=batch['input_ids'],
                max_horizon_length=args.regression.Time_MOE.inference_model_args.prediction_length,
                return_dict=True,
                **{key: batch[key] for key in batch if key.startswith('channel_')}
            )
            
            #print(f'Output Shape: {output.logits.shape}')
            forecast = output.logits[:, -1, :]   # [B, prediction_len, F] simplified

            #print(f'Forecast Shape: {forecast.shape}')
            
            # --- Inverse scaling per subsequence ---
            forecast_original_scale = []
            for i in range(forecast.shape[0]):
                f = forecast[i]                                # [prediction_len, F]
                mean = batch['mean'][i]           # [1, F]
                std = batch['std'][i]             # [1, F]
                f_inv = inference_loader.dataset.inverse_scale(f, mean, std)
                forecast_original_scale.append(f_inv)

            forecast_original_scale = torch.stack(forecast_original_scale, dim=0)  # [B, prediction_len, F]

            # Store in batch
            batch['model_prediction_sequence'] = forecast_original_scale
            all_sequences.append(batch)
            
    logging.info(f"Model Inferencing Completed!")
    
    if args.regression.S3.save_sequences:
        seq_path = args.regression.S3.output_seq_path
        logging.info(f'Saving the individual inferenced sequences')
        
        # Saving the all_sequences as a whole
        with open(seq_path, "wb") as f:
            pickle.dump(all_sequences, f)
        logging.info(f'Saved the individual inferenced sequences to {seq_path}')
    return all_sequences
