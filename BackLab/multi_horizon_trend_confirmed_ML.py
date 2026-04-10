from turtle import pd
import preloads.load_all
import logging

from pathlib import Path
from src.log import Log
from src.helper import timeit
from src.backtest_handler import BacktestHandler
from src.data_handler import DataHandler
from objs.inputs import Inputs
from datetime import datetime as dt
import time
import numpy as np
from indicator.sma import SMA
import sys
import os
from pathlib import Path
import pandas as pd

# Get project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ALSO add ML_Core/src to path for internal imports
ml_core_src = project_root / "ML_Core" / "src"
if str(ml_core_src) not in sys.path:
    sys.path.insert(0, str(ml_core_src))


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


from ML_Core.src.utils.utils import load_config, config_to_args
import torch
from ML_Core.src.time_moe.models.modeling_time_moe import TimeMoeForPrediction
from ML_Core.src.time_moe.datasets.time_moe_dataset import MultiFrequencyDatasetLoader


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



class Optimization:
    def __init__(self):
        self.run = False

        # example
        self.params = {'a': [0, 1, 2, 3],
                       'b': [6, 7, 8, 9]}

class Data:
    def __init__(self, log):

        ticker_list_path = "BackLab/data/tickers_448_v2.txt"
        with open(ticker_list_path, 'r') as f:
            self.tickers = f.read().splitlines()

        logging.info(f"Loaded {len(self.tickers)} tickers for backtest.")
        data_handler = DataHandler(self.tickers, start_date = "2010-04-01", end_date = "2025-01-01", interval = '1d',
                                   reference_ticker = "^STI", method = "parquet", log = log)

        # dont touch
        self.data_handler = data_handler
        self.stock_data = data_handler.stock_data
        self.reference_data = data_handler.reference_data

class BacktestLogic:
    """
    Multi-Horizon Trend-Confirmed Strategy
    Uses the MC-MoE model's 8-step forecast to require BOTH short-term (day+1)
    AND medium-term (day+8) positive price predictions before investing.
    - Filters out tickers where only the immediate next day looks positive but the medium-term trend is negative, reducing noise-driven entries.

    Forecast tensor shape: [8] — predicted prices for day+1 through day+8
      forecast[0] = predicted price for day+1 (short-term signal)
      forecast[7] = predicted price for day+8 (medium-term trend filter)
    """
    def __init__(self, parameters = None):
        self.filename = os.path.basename(__file__)
        self.inputs = Inputs(initial_capital = 100000, leverage = 1, rebalance_proportion_diff = 0.10, commission = {"fix_per_trade": 0.0, "percent": 0.00825},
                    slippage = 0.001, slippage_type = "percent", shorting_cost = 0.00, profit_taking = 0,
                    stop_loss = 0, min_reentry_bar = 0, price_filter = 0.001, rebalance_on_bar_open = True, rebalance_on_bar_close = False,
                    create_log = False, create_performance_file = True, snapshot = True, filename = self.filename)

        # Loading the args
        config_file_path = 'ML_Core/config/config_tickers_448_7_Channels_with_temporal_tape_v2.yaml'
        config_dict = load_config(config_file_path)

        place_holders_to_replace = {'version_name': config_dict['regression']['common']['version_name']}
        args = config_to_args(config_dict, place_holders_to_replace)
        logging.info(f"Current Config Version Name: {args.regression.common.version_name}")


        # Loading the model
        device = 'cpu'
        logging.info(f"Loading Model from HuggingFace @ {args.regression.Time_MOE.model_path}")
        model = TimeMoeForPrediction.from_pretrained(
            args.regression.Time_MOE.model_path,
            device_map='cpu' if not torch.cuda.is_available() else ('auto' if device is None else device),
            torch_dtype='auto',
        )

        self.args = args
        self.model = model

        self.prep_all_tensors_from_feature_engineered_df()

        return


    def prep_all_tensors_from_feature_engineered_df(self):
        '''
        Prepping the DataLoader from the Feature Engineered DataFrame
        '''
        feature_engineered_df = pd.read_parquet('ML_Core/data/processed_data/feature_engineered_data_tickers_448_7_Channels_with_temporal_tape_v2.parquet')
        logging.info(f'''Preparing DataLoader from Feature Engineered DataFrame with shape: {feature_engineered_df.shape}''')

        self.dataloader = MultiFrequencyDatasetLoader(
            args=self.args,
            stage='full',
            feature_engineered_df=feature_engineered_df,
            shuffle=False,
            batch_size=16,
            num_workers=1,
            context_length=self.args.regression.Time_MOE.train_model_args.max_length,
            prediction_length=self.args.regression.Time_MOE.inference_model_args.prediction_length,
        )
        logging.info(f'DataLoader prepared!')


    def model_predict(self, input_tensor):
        '''
        Input Tensor for Model Prediction
        '''
        with torch.no_grad():
            output = self.model(
                input_ids=input_tensor['input_ids'],
                max_horizon_length=self.args.regression.Time_MOE.inference_model_args.prediction_length,
                return_dict=True,
                **{key: input_tensor[key] for key in input_tensor if key.startswith('channel_')}
            )

            forecast = output.logits[:, -1, :]   # [B, 8] — 8 predicted future prices
            tickers = input_tensor['ticker']  # [B]

            # --- Inverse scaling per subsequence ---
            forecast_original_scale = []
            for i in range(forecast.shape[0]):
                f = forecast[i]                                # [8]
                mean = input_tensor['mean'][i]           # [1, F]
                std = input_tensor['std'][i]             # [1, F]
                f_inv = inverse_scale(f, mean, std)
                forecast_original_scale.append(f_inv)

            forecast_original_scale = torch.stack(forecast_original_scale, dim=0)  # [B, 8]

            return forecast_original_scale, tickers

    def prep_tensor_for_prediction(self, date_time, batch_size=16):
        '''
        Receiving date_time and then retrieving the data
        '''
        input_lists = self.dataloader.dataset.getItemByDate(date_time)
        if len(input_lists) == 0:
            logging.warning(f'No data available for date_time: {date_time}. Skipping prediction.')
            return []
        else:
            logging.info(f'Preparing input tensor for date_time: {date_time} with {len(input_lists)} samples')
            # Collate into batches
            batched_inputs = []
            for i in range(0, len(input_lists), batch_size):
                batch = MultiFrequencyDatasetLoader.collate_multi_frequency_fn(input_lists[i:i+batch_size])
                batched_inputs.append(batch)

            return batched_inputs

    def on_bar_open(self, stocks, bar, bar_type, date_time, date_time_series):

        # [open] is only updated at on_bar_open whereas [high/low/close] are not updated
        logging.info(f'Date Time: {date_time}, Bar: {bar}, Bar Type: {bar_type}')
        batched_inputs = self.prep_tensor_for_prediction(date_time, batch_size=32)

        # Input to model for forecasting
        all_sequence_forecasts, all_sequence_tickers = [], []
        for batch in batched_inputs:
            batch_forecast, batch_tickers = self.model_predict(batch)  # [B, 8]
            all_sequence_forecasts.extend(batch_forecast)
            all_sequence_tickers.extend(batch_tickers)

        assert len(all_sequence_forecasts) == len(all_sequence_tickers), "Mismatch in number of forecasts and tickers"

        # Trend-Confirmed Strategy: require BOTH day+1 AND day+8 to predict price increase
        trend_confirmed_tickers = []
        for i in range(len(all_sequence_forecasts)):
            forecast = all_sequence_forecasts[i]  # [8] — predicted prices for day+1 through day+8
            ticker = all_sequence_tickers[i]
            current_price = stocks[ticker].close[0]

            predicted_day1 = forecast[0].item()   # day+1 predicted price
            predicted_day8 = forecast[7].item()   # day+8 predicted price

            short_term_positive = predicted_day1 > current_price
            medium_term_positive = predicted_day8 > current_price

            if short_term_positive and medium_term_positive:
                trend_confirmed_tickers.append(ticker)

        logging.info(f'Trend-confirmed tickers for {date_time}: {len(trend_confirmed_tickers)}')
        if (bar > 0) and len(trend_confirmed_tickers) > 0:
            for ticker, o in stocks.items():
                o.proportion = 1/len(trend_confirmed_tickers) if ticker in trend_confirmed_tickers else 0
        return stocks

    def on_bar_close(self, stocks, bar, bar_type, date_time, date_time_series):
        # [high/low/open/close] are updated at on_bar_close

        return stocks



@timeit
def main():
    backtest_handler = BacktestHandler(BacktestLogic, Data, Optimization)
    backtest_handler.run()
    return

if __name__ == "__main__":
    main()
