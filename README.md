# MC-MoE: Multi-Channel Mixture-of-Experts for Financial Time Series Forecasting

A custom-built **multi-variate time series forecasting model** that extends the [TimeMoE-50M](https://huggingface.co/Maple728/TimeMoE-50M) architecture with multi-channel fundamental data fusion, temporal position encoding, and a Mixture-of-Experts (MoE) decoder. It predicts short-horizon stock open prices across a universe of ~448 tickers using 7 heterogeneous data channels fused via cross-attention.

---

## Architecture Overview

MC-MoE augments the original TimeMoE decoder-only transformer with a **Multi-Frequency Input Embedding** layer (`MultiFreqInputEmbedding`) that fuses price data with lower-frequency fundamental and alternative data channels before the transformer backbone processes them.

```
Input Channels
│
├── Channel 1 (Price): Close price time series [63 timesteps]
│   └── LSTM(1 → 384)
│
├── Channel 2 (DuPont): Net margin, asset turnover, equity multiplier, ROE [6 timesteps]
│   └── TimeEncodedLSTM(4 → 384) + tAPE
│
├── Channel 3 (Quality): Cash generation, growth, leverage, profitability, score [6 timesteps]
│   └── TimeEncodedLSTM(5 → 384) + tAPE
│
├── Channel 4 (Sentiment): Social sentiment score [10 timesteps]
│   └── TimeEncodedLSTM(1 → 384) + tAPE
│
├── Channel 5 (Insider vol): Change in insider transaction volume [5 timesteps]
│   └── TimeEncodedLSTM(1 → 384) + tAPE
│
├── Channel 6 (Gov spending): USA contract spending [5 timesteps]
│   └── TimeEncodedLSTM(1 → 384) + tAPE
│
└── Channel 7 (Insider flow): Share purchase ratio + net insider change [5 timesteps]
    └── TimeEncodedLSTM(2 → 384) + tAPE
│
▼
Cross-Attention Fusion (MultiheadAttention, 4 heads)
Query: price embedding | Keys/Values: channel embeddings (interpolated to T)
│
▼
TimeMoE Transformer Backbone (12 layers, hidden=384)
  Each layer: Causal Self-Attention + Sparse MoE FFN (8 experts, top-2 routing)
│
▼
Multi-Horizon Output Heads: [1-step, 8-step, 32-step, 64-step]
│
▼
Predicted next_open prices
```

**Key design decisions:**

| Component | Detail |
|---|---|
| **Backbone** | TimeMoE-50M (12-layer decoder, 384 hidden, 8 MoE experts per layer) |
| **Channel fusion** | Cross-attention between price query and interpolated channel keys/values |
| **Temporal encoding** | Temporal Awareness Positional Encoding (tAPE) — frequency-corrected sinusoidal PE scaled by actual sequence length and days-from-end timestamps |
| **Loss** | HuberLoss (δ=2.0) on the first feature only (Open price) |
| **Aux loss** | Load-balancing loss from Switch Transformer (factor 0.02) |
| **Output** | Multi-horizon heads simultaneously predict 1, 8, 32, 64 steps ahead |
| **Parameters** | ~130M |

---

## Data Sources

| Dataset | Source | Update Frequency |
|---|---|---|
| OHLCV prices | Yahoo Finance | Daily |
| DuPont ratios | FMP Financials API | Quarterly |
| Earnings quality scores | FMP / Finnhub | Quarterly |
| Social sentiment | Finnhub | Daily |
| Insider transactions | FMP / Finnhub | Event-driven |
| Insider sentiment (MSPR) | Finnhub | Monthly |
| USA government spending | Finnhub | Event-driven |
| Historical ESG scores | Finnhub | Annual |
| Earnings surprises | Finnhub / FMP | Quarterly |

The ticker universe covers **448 tickers** — primarily S&P 500 constituents plus selected international stocks. Tickers with incomplete history before 2010 are excluded.

---

## Repository Structure

```
ML_Core/
├── config/
│   ├── config_tickers_448_7_Channels_with_temporal_tape.yaml    # main training config
│   └── config_tickers_448_7_Channels_with_temporal_tape_v2.yaml # v2 with FMP data
├── data/
│   ├── raw_data/          # downloaded parquets (OHLCV, fundamentals, etc.)
│   ├── processed_data/    # feature-engineered parquets, pkl sequences
│   └── outputs/           # inference results, tensorboard images
├── src/
│   ├── S1_Data_Preprocessing.py   # data ingestion + alt data merging
│   ├── S2_Feature_Engineering.py  # DuPont ratios, Yeo-Johnson transforms
│   ├── S3_Model_Training.py       # dataset prep + inference runner
│   ├── S4_Model_Validation.py     # regression metrics, plots
│   ├── S5_Model_Training_Evaluation.py  # training curve plots
│   ├── scenario_analysis.ipynb    # Monte Carlo scenario analysis
│   ├── time_moe_data_preprocessing.ipynb
│   ├── time_moe_model_training.ipynb
│   ├── time_moe_model_training.py         # CLI training script
│   ├── time_moe_model_training_with_optuna.py  # Optuna HPO script
│   ├── time_moe_features_inferencing.ipynb    # augmented inference
│   ├── models/
│   │   ├── mlp_dummy.py
│   │   ├── model_trainer.py
│   │   ├── time_moe_wrapper.py
│   │   └── transformers_model_trainer.py
│   ├── time_moe/
│   │   ├── datasets/
│   │   │   ├── time_moe_dataset.py    # core dataset classes (V2, V3, loaders)
│   │   │   ├── binary_dataset.py
│   │   │   └── general_dataset.py
│   │   ├── models/
│   │   │   ├── modeling_time_moe.py   # full model implementation
│   │   │   └── configuration_time_moe.py
│   │   ├── runner.py                  # TimeMoeRunner (train + infer)
│   │   └── trainer/hf_trainer.py     # custom HF Trainer subclass
│   └── utils/
│       ├── utils.py
│       ├── data_download.ipynb
│       ├── data_download_fmp.ipynb
│       ├── feature_engineering_fmp.ipynb
│       └── finnhub_crawl.py / finnhub_crawl_2.py

```

---

## Prerequisites

### Environment Setup

The project uses separate Python environments for different model families. Install the primary training environment:

```bash
# Python 3.12 recommended
pip install torch transformers datasets huggingface_hub
pip install pandas numpy scipy scikit-learn optuna
pip install finnhub-python yfinance certifi
pip install tensorboard matplotlib seaborn
pip install python-dotenv tqdm fastparquet
```

### Environment Variables

Create `ML_Core/config/.env`:

```env
FINNHUB_API_KEY=your_finnhub_key
HF_TOKEN=your_huggingface_write_token
FMP_API_KEY=your_fmp_key          # for v2 data pipeline
```

---

## End-to-End Training Pipeline

The pipeline has six sequential stages (S1–S6). Each writes its output to `data/` for the next stage to consume.

### Stage 0 — Data Download

**Download OHLCV prices:**

```bash
# Run data/download.ipynb or:
python -c "
import yfinance as yf, pandas as pd
tickers = open('data/raw_data/tickers_448_v2.txt').read().splitlines()
data = yf.download(tickers, start='2010-01-01', end='2025-01-01', group_by='ticker')
# ... (see utils/data_download.ipynb for full code)
"
```

**Download alternative data (Finnhub + FMP):**

```bash
# Async crawler for insider transactions, sentiment, ESG, earnings, USA spending:
python src/utils/finnhub_crawl_2.py

# FMP financial statements, ratings, analyst grades:
# Run src/utils/data_download_fmp.ipynb
```

All raw data lands in `data/raw_data/` as `.parquet` files.

---

### Stage 1 — Data Preprocessing (S1)

Merges OHLCV with all alternative data sources on `[Date, Ticker]`.

Runs automatically as part of the notebook pipeline. To run standalone:

```python
from src.utils.utils import load_config, config_to_args
from src.S1_Data_Preprocessing import S1_preprocessing

config = load_config("config/config_tickers_448_7_Channels_with_temporal_tape_v2.yaml")
args = config_to_args(config, {'version_name': config['regression']['common']['version_name']})

full_df = S1_preprocessing(args)
# Saves to: data/processed_data/processed_data_{version_name}.parquet
```

---

### Stage 2 — Feature Engineering (S2)

Computes DuPont ratios, quality scores, optional Yeo-Johnson transforms, and the `next_open` target label.

**For the Finnhub pipeline (original config):**

```python
from src.S2_Feature_Engineering import S2_feature_engineering
feature_df = S2_feature_engineering(args, full_df)
# Saves to: data/processed_data/feature_engineered_data_{version_name}.parquet
```

**For the FMP pipeline (v2 config):** Run `src/utils/feature_engineering_fmp.ipynb` which computes DuPont ratios, cash flow efficiency, leverage, analyst scores, and rating ranks from FMP data, then saves `feature_engineered_data_tickers_448_7_Channels_with_temporal_tape_v2.parquet`.

---

### Stage 3 — Model Training

**Option A: Jupyter Notebook (interactive)**

Open `src/time_moe_model_training.ipynb` and run all cells. It will:
1. Load the config
2. Initialize `TimeMoeRunner`
3. Run training with or without Optuna HPO
4. Save model to `logs/time_moe_{version_name}/`
5. Upload to HuggingFace Hub
6. Run inference and save sequences to `data/outputs/`

**Option B: CLI (standard training)**

```bash
cd ML_Core/src
python time_moe_model_training.py
```

**Option C: CLI with Optuna hyperparameter search**

```bash
cd ML_Core/src
python time_moe_model_training_with_optuna.py
```

This runs `n_trials=20` Optuna trials searching over:
- Learning rate: [1e-4, 1e-3]
- Weight decay: [0.0, 0.1]
- Epochs: [2, 3]
- Warmup steps: [0, 1000]
- Batch size: [8, 16, 32]

After HPO it automatically retrains on the full dataset with the best hyperparameters and pushes to HuggingFace.

**Key training configuration** (from `config_tickers_448_7_Channels_with_temporal_tape_v2.yaml`):

```yaml
train_model_args:
  input_size: 42                # total input feature dimensions
  prediction_length: 8          # forecast 8 trading days ahead
  max_length: 63                # 63-day context window (~3 months)
  normalization_method: "zero"  # z-score normalisation per subsequence
  num_train_epochs: 2.0
  learning_rate: 0.0001
  per_device_train_batch_size: 16
  global_batch_size: 128        # gradient accumulation steps = 8
  precision: 'fp32'
  torch_compile: True
  lr_scheduler_type: "cosine"
  weight_decay: 0.1
  optim: "adamw_torch_fused"
  early_stopping_patience: 3
```

The dataset resampling weights by volatility bucket allow oversampling mid-to-high volatility tickers during training:

```yaml
bucket_mapping_jsons:
  volatility:
    weights: {'0': 0.1, '1': 0.5, '2': 1.5, '3': 1.4, '4': 1.5}
```

---

### Stage 4 — Model Validation (S4)

Runs automatically after inference in the training notebook. Produces:
- Overall regression metrics (RMSE, MAE, MAPE, R², Bias, Correlation)
- Per-ticker breakdowns
- Metrics stratified by volatility bucket
- Residual plots and predicted vs actual visualisations

```python
from src.S4_Model_Validation import S4_Model_Validation

regression_res, seq_df, ticker_df, result_df, result_no_dupe, result_std = \
    S4_Model_Validation(args, all_sequences_inf)
```

---

### Stage 5 — Training Curve Evaluation (S5)

Parses TensorBoard event logs and plots train/val loss, gradient norms, and learning rate curves. Outputs are saved to `data/outputs/{version_name}/training_metrics.png`.

---

### Stage 6 — Monte Carlo Scenario Analysis (S6)

Generates augmented inference by systematically mutating fundamental feature channels to historical quantile values, producing scenario-level forecasts.

**Run augmented inference:**

```bash
# Open src/time_moe_features_inferencing.ipynb
# Uses MultiFrequencyDatasetLoaderV2 with augmentation_config from S6 config block
```

**Run scenario analysis notebook:**

```bash
# Open src/scenario_analysis.ipynb
# Produces:
#   - Fan charts per ticker (5th–95th percentile prediction bands)
#   - Feature sensitivity ranking (which fundamentals move forecasts most)
#   - 90% confidence interval width by forecast horizon
#   - Scenario-based VaR and CVaR at each horizon
```

Features mutated in the v2 config:
```yaml
S6:
  features_to_mutate:
    - ["dupont_net_margin", 1008]          # 4-year lookback
    - ["cashGenerationCapitalAllocation", 1008]
    - ["growth", 1008]
    - ["leverage", 1008]
    - ["profitability", 1008]
    - ["score", 1008]
    - ["rating_rank", 504]                 # 2-year lookback
  start_date: '2020-01-01'
  end_date: '2021-01-01'
```

---

## HuggingFace Model

The trained model is published to:

```
kiatkock/MV-Time-MOE
```

Load for inference:

```python
import torch
from src.time_moe.models.modeling_time_moe import TimeMoeForPrediction
from src.time_moe.datasets.time_moe_dataset import MultiFrequencyDatasetLoader

model = TimeMoeForPrediction.from_pretrained(
    "kiatkock/MV-Time-MOE",
    device_map="auto",
    torch_dtype=torch.float32,
)

# Build your MultiFrequencyDatasetLoader from a feature-engineered DataFrame
# then iterate and call model(input_ids=..., channel_1=..., channel_1_dates=..., ...)
```

---

## Configuration Reference

The YAML configs expose all pipeline knobs. Key sections:

```
regression:
  common:      version_name, unique_keys
  S1:          ticker list, raw data paths, date range, tickers to exclude
  S2:          target label (next_open), feature engineering options
  S3:          train/test/inference date splits, window_size=63, channel_configs
  S4:          validation subsets, plotting flags
  Time_MOE:    model_path, output_path, Optuna grid, train/inference model args
  S6:          scenario analysis features, lookback windows, date range
```

Channel configs format: `[patch_length, stride, num_features]`

```yaml
channel_configs:
  - [63, 1, 1]   # price: 63-step window, 1 feature
  - [6,  1, 4]   # DuPont: 6 quarterly observations, 4 features
  - [6,  1, 5]   # quality scores: 6 obs, 5 features
  - [10, 1, 1]   # social sentiment: 10 obs, 1 feature
  - [5,  1, 1]   # insider volume: 5 obs, 1 feature
  - [5,  1, 1]   # government spending: 5 obs, 1 feature
  - [5,  1, 2]   # insider flow: 5 obs, 2 features
```

---

```
