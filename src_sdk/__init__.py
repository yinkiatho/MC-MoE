"""
src_sdk — public SDK for Time-MoE stock-market forecasting.

Typical usage::

    from src_sdk import TimeMoeSDK, ModelOutput

    # Inference from HuggingFace Hub
    sdk = TimeMoeSDK.from_pretrained("kiatkock/MV-Time-MOE")
    result = sdk.predict(input_batch, max_horizon_length=1)

    # Fine-tuning
    sdk = TimeMoeSDK(model_path="kiatkock/MV-Time-MOE", output_path="./runs")
    sdk.train_model(config_path="config/train.yaml")
"""

from .TimeMoeSDK import TimeMoeSDK
from .BaseConfig import ModelOutput
from .BaseModel import BaseModel
from .time_moe_dataloader import (
    TimeMoeDataLoader,
    TimeMoeTorchDataset,
    create_dataloader,
    collate_multi_frequency_fn,
)

__all__ = [
    "TimeMoeSDK",
    "ModelOutput",
    "BaseModel",
    "TimeMoeDataLoader",
    "TimeMoeTorchDataset",
    "create_dataloader",
    "collate_multi_frequency_fn",
]
