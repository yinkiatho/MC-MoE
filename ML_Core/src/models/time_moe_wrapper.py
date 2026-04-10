import torch
import torch.nn as nn
from argparse import Namespace
from transformers import AutoModelForCausalLM


class Time_MOE_Wrapper(nn.Module):
    def __init__(
        self,
        args: Namespace,
    ):
        super().__init__()

        # Load Time-MoE (no tokenizer)
        self.time_moe = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-50M",
            device_map="cpu",
            trust_remote_code=True,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,                 # shape [B, context_len], float
        prediction_length: int = 1,
        assume_normalized: bool = False  # if False, we z-score then inverse at the end
    ):
        """
        Returns:
            forecasts: shape [B, prediction_length] if no projection,
                       else shape [B, output_dim]
        """
        assert x.dim() == 2, "x must be [batch, context_len] float tensor"

        device = next(self.time_moe.parameters()).device
        x = x.to(device)

        # Normalize if requested
        if assume_normalized:
            normed = x
            mean, std = None, None
        else:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp_min(1e-8)
            normed = (x - mean) / std

        # Autoregressive forecasting via generate
        # Output shape: [B, context_len + prediction_length]
        out = self.time_moe.generate(
            normed,
            max_new_tokens=prediction_length
        )
        normed_forecast = out[:, -prediction_length:]  # [B, prediction_length]

        # Inverse normalization if we normalized
        if not assume_normalized:
            forecast = normed_forecast * std + mean
        else:
            forecast = normed_forecast

        return forecast
    
    
    def generate(self, input_sequence: torch.Tensor, max_new_tokens: int):
        '''
        Multi-Resolution using the .generate from Time-MoE
        '''
        output = self.time_moe.generate(input_sequence, max_new_tokens=max_new_tokens)  # shape is [batch_size, 12 + 6]
        normed_predictions = output[:, -max_new_tokens:]
        return output 
        
