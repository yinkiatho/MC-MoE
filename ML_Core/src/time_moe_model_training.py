import os
import pandas as pd
from dotenv import load_dotenv
from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
import datetime



import logging
from time_moe.datasets import *
from time_moe.runner import *
from time_moe.trainer import *
from time_moe.utils import *
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from huggingface_hub import HfApi
from datetime import datetime, timezone, timedelta
from S3_Model_Training import S3_model_inferencing_multi_frequency
import torch
from utils.utils import patch_torch_load
patch_torch_load()


# Clear handlers to prevent duplicate log entries from imported modules
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(    
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config/.env'))

logging.info(f"CWD: {os.getcwd()}")
os.chdir(os.path.dirname(os.path.abspath(__file__)))



# Check what GPUs PyTorch sees
print(f'Number of GPUs Available: {torch.cuda.device_count()}')  # should show 1 if you set "1"
print(f'Device Name: {torch.cuda.get_device_name(0)}')  # the GPU PyTorch will use
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}')

# Check actual GPU being used
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'Current CUDA device: {device}')
    print(f'Device properties: {torch.cuda.get_device_properties(device)}')

# Loading of config
from utils.utils import load_config, config_to_args

config_file_path = "../config/config_tickers_448_7_Channels_with_temporal_tape.yaml"
config_dict = load_config(config_file_path)

place_holders_to_replace = {'version_name': config_dict['regression']['common']['version_name']}
args = config_to_args(config_dict, place_holders_to_replace)

print(f"Current Version Name: {args.regression.common.version_name}")

runner = TimeMoeRunner(
        model_path=args.regression.Time_MOE.model_path,
        output_path=args.regression.Time_MOE.output_path,
        seed=args.regression.Time_MOE.train_seed,
    )

train_model_params_dict = args.regression.Time_MOE.train_model_args.__dict__
train_model_params_dict['args'] = args
    
model, _ = runner.train_model(
    **train_model_params_dict
)

# Tensor Board 
# Path to your events file
event_file = Path(f"{args.regression.Time_MOE.output_path}/tb_logs")
logging.info(f"Searching for TB Logs @ {event_file}")
latest_event = [i for i in os.listdir(event_file) if i.startswith('events.out.tfevents')][-1]
logging.info(f"Found TensorBoard Event File {latest_event}")
file_path = os.path.join(event_file, latest_event)

# Lists to store results
steps = []
tags = []
values = []

# Iterate through the event file
for e in tf.compat.v1.train.summary_iterator(str(file_path)):
    for v in e.summary.value:
        steps.append(e.step)
        tags.append(v.tag)
        values.append(v.simple_value)

# Convert to DataFrame for easy analysis
df = pd.DataFrame({
    "step": steps,
    "tag": tags,
    "value": values
}).drop_duplicates()

# Assume df is your DataFrame
df_filtered = df[df['step'] != 0]  # ignore step 0

# Pivot with aggregation to handle duplicates
df_pivot = df_filtered.pivot_table(
    index='step',
    columns='tag',
    values='value',
    aggfunc='first'  # take the first occurrence if duplicates exist
).reset_index()


metrics = ['train/loss', 'train/grad_norm', 'train/learning_rate', 'eval/loss']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 layout
axes = axes.flatten()  # flatten into 1D array for easy looping

for i, m in enumerate(metrics):
    ax = axes[i]
    df_pivot_metrics = df_pivot[['step', m]].dropna()
    ax.plot(df_pivot_metrics['step'], df_pivot_metrics[m], label=m)
    ax.set_title(f'{m}', fontsize=14)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig(args.regression.S3.tensorboard_diagram, dpi=300, bbox_inches='tight')


# Pushing to HuggingFace
# Define Singapore timezone (UTC+8)
sgt = timezone(timedelta(hours=8))
datetime_str = datetime.now(sgt).strftime("%Y-%m-%d %H:%M:%S")

# Generate model card before upload
def generate_model_card(args, output_path):
    train_args = args.regression.Time_MOE.train_model_args
    card = f"""---
            language: en
            tags:
            - time-series
            - financial-forecasting
            - mixture-of-experts
            license: mit
            ---
            # MV-Time-MOE ({args.regression.common.version_name})

            Multi-variate Time-series Mixture-of-Experts model for financial forecasting.

            ## Model Details
            - **Input size**: {train_args.input_size}
            - **Prediction length**: {train_args.prediction_length}
            - **Max context length**: {train_args.max_length}
            - **Channel configs**: {train_args.channel_configs}

            ## Training
            - Learning rate: {train_args.learning_rate}
            - Epochs: {train_args.num_train_epochs}
            - Batch size (micro): {train_args.micro_batch_size}
            - Precision: {train_args.precision}
            - Optimizer: {train_args.optim}
            - Weight decay: {train_args.weight_decay}
            - LR scheduler: {train_args.lr_scheduler_type}
            - Trained on: {datetime_str}

            ## Usage
            ```python
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction

            model = TimeMoeForPrediction.from_pretrained("{args.regression.Time_MOE.model_path}")
            ```
            """
    with open(os.path.join(output_path, "README.md"), "w") as f:
        f.write(card)
    logging.info(f"Generated model card at {output_path}/README.md")

generate_model_card(args, args.regression.Time_MOE.output_path)

# Initialize API and upload
api = HfApi(token=os.environ.get("HF_TOKEN"))

try:
    api.upload_folder(
        folder_path=args.regression.Time_MOE.output_path,
        repo_id=args.regression.Time_MOE.model_path,
        repo_type="model",
        commit_message=f"{args.regression.common.version_name} model training @ {datetime_str}",
        ignore_patterns=["tb_logs/*", "checkpoint-*", "runs/*", "*.log"],
    )
    logging.info(f"Successfully uploaded model to {args.regression.Time_MOE.model_path}")
except Exception as e:
    logging.error(f"HuggingFace upload failed: {e}")
    raise



# Inferencing

# Model Inferencing Code
all_sequences_inf = S3_model_inferencing_multi_frequency(args, None, None)