import pandas as pd
import numpy as np
import json
import datetime
from argparse import Namespace
import logging
import scipy
import statistics
from scipy.stats import boxcox, yeojohnson

def apply_transformation(group, col, method='yeo_johnson', lambda_val=None):
    """
    Box-Cox or Yeo-Johnson to a single column of a ticker group.
    Returns transformed column and the estimated lambda 
    """
    data = group[col].dropna().values   # avoid NaNs in transformation
    
    if method == 'box_cox':
        if lambda_val is None:
            transformed, lam = boxcox(data)   # lam is the estimated lambda
        else:
            transformed = boxcox(data, lmbda=lambda_val)
            lam = lambda_val
    elif method == 'yeo_johnson':
        if lambda_val is None:
            transformed, lam = yeojohnson(data)
        else:
            transformed = yeojohnson(data, lmbda=lambda_val)
            lam = lambda_val
    else:
        raise ValueError("method must be 'box_cox' or 'yeo_johnson'")
    
    # Return series aligned with original index (non‑NaN positions)
    transformed_series = pd.Series(index=group[col].dropna().index, data=transformed)
    return transformed_series, lam


def S2_feature_engineering(args: Namespace, full_df: pd.DataFrame) -> pd.DataFrame:    
    '''
    Main entry function to feature engineering S2, returns the feature engineered data frame
    '''
    logging.info(f"Initiating S2 Feature Engineering with {len(full_df)} rows currently...............")
    
    feature_engineered_df = feature_engineering(args, full_df)
    if args.regression.S2.save_file:
        feature_engineered_df.to_parquet(args.regression.S2.feature_engineered_file_path)
        logging.info(f"Saved feature engineered dataframe to {args.regression.S2.feature_engineered_file_path}")
        
    logging.info(f"Completed S2 Feature Engineering...............")
    return feature_engineered_df


def add_indicators(group):
    '''
    Adds technical indicators to a single ticker group: SMA_5, SMA_20, Vol_20 (rolling std),
    RSI_14, and Return_5 (5-day pct change).
    '''
    
    # 5-day and 20-day simple moving averages
    group['SMA_5'] = group['Close'].rolling(5).mean()
    group['SMA_20'] = group['Close'].rolling(20).mean()
    
    # 20-day rolling std (volatility)
    group['Vol_20'] = group['Close'].rolling(20).std()
    
    # 14-day RSI
    delta = group['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    group['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 5-day rolling return
    group['Return_5'] = group['Close'].pct_change(5)
    return group


def feature_engineering(args: Namespace, full_df: pd.DataFrame, use_indicators=False):
    # 1. Sort and create technical indicators
    full_df = full_df.sort_values(by=['Ticker', 'Date'])

    if use_indicators:
        feature_engineered_df = full_df.groupby('Ticker', group_keys=False).apply(add_indicators)
    else:
        feature_engineered_df = full_df.copy()
    
    # 2. Apply chosen transformation to specified columns
    transform_cols = args.regression.S2.transform_cols
    method = args.regression.S2.transform_method
    lambda_fixed = args.regression.S2.lambda_param
    
    # Store lambdas per ticker and column (if needed)
    lambda_store = {}   # dict of {(ticker, col): lam}

    for col in transform_cols:
        if col not in feature_engineered_df.columns:
            logging.warning(f"Column {col} not found – skipping transformation")
            continue
        
        # Apply per ticker group
        transformed_list = []
        lambda_list = []
        for ticker, group in feature_engineered_df.groupby('Ticker'):
            trans_series, lam = apply_transformation(group, col, method, lambda_fixed)
            transformed_list.append(trans_series)
            lambda_list.append((ticker, col, lam))
            if args.regression.S2.store_lambda:
                lambda_store[(ticker, col)] = lam
        
        # Combine transformed series and replace original column
        if transformed_list:
            all_transformed = pd.concat(transformed_list).sort_index()
            feature_engineered_df[col] = all_transformed
    
    feature_engineered_df.attrs['lambda_store'] = lambda_store
    feature_engineered_df = feature_engineered_df.dropna().reset_index(drop=True)
    return feature_engineered_df

