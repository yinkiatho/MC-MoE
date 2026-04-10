import pandas as pd
import numpy as np
import json
import datetime
from argparse import Namespace
import logging
import scipy
import statistics
from utils.utils import DataFrameIterator
import torch
from models.mlp_dummy import MLPModel
from models.model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import os


def S5_Model_Training_Evaluation(args: Namespace, best_train_log: dict):
    '''
    S5 Module to plot out Training Logs for model training
    '''
    logging.info(f"Initialising the Model Training Logs Evaluation.......................")
    train_loss, val_loss = best_train_log['Train Losses'], best_train_log['Val Losses']
    step_train_loss, step_val_loss = best_train_log['Step Train Losses'], best_train_log['Step Val Losses']
    step_grad_norms = best_train_log['Step Gradient Norms']
    
    
    # Plotting these functions
    plot_training_metrics(args, train_loss, val_loss, step_train_loss, step_val_loss, step_grad_norms)
    logging.info(f"Plotted Training logs plot")
    return 
    
def plot_training_metrics(args, train_losses, val_losses, train_step_losses, val_step_losses, grad_norms):
    output_dir = f'../data/outputs/{args.regression.common.version_name}/'
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # ---- Row 1: Epoch losses ----
    axes[0, 0].plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", color="blue")
    axes[0, 0].set_title("Train Loss (per epoch)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(range(1, len(val_losses)+1), val_losses, label="Val Loss", color="orange")
    axes[0, 1].set_title("Validation Loss (per epoch)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    # ---- Row 2: Step losses ----
    axes[1, 0].plot(range(1, len(train_step_losses)+1), train_step_losses, label="Train Step Loss", color="green")
    axes[1, 0].set_title("Train Loss (per step)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].plot(range(1, len(val_step_losses)+1), val_step_losses, label="Val Step Loss", color="red")
    axes[1, 1].set_title("Validation Loss (per step)")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Loss")

    # ---- Row 3: Grad norm ----
    axes[2, 0].plot(range(1, len(grad_norms)+1), grad_norms, label="Grad Norm", color="purple")
    axes[2, 0].set_title("Gradient Norm (per step)")
    axes[2, 0].set_xlabel("Step")
    axes[2, 0].set_ylabel("Norm")

    # Hide the unused subplot (bottom right)
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300)
    plt.show() 
    plt.close()