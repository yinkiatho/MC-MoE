import os
import math
import random
from functools import reduce
from operator import mul
import pandas as pd
import torch
from transformers import EarlyStoppingCallback
from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
from time_moe.utils.dist_util import get_world_size
from time_moe.utils.log_util import logger, log_in_local_rank_0
from time_moe.datasets.time_moe_dataset import MultiFrequencyDatasetLoader, MultiFrequencyTimeSeriesDataset, TimeMoEMultiFrequencyWindowDataset, TimeMoEMultiFrequencyWindowDatasetV2, collate_multi_frequency_fn
import json
from argparse import Namespace

try:
    from src_sdk.BaseModel import BaseModel
    from src_sdk.BaseConfig import ModelOutput
except ImportError:
    # Fallback stubs so runner works standalone when src_sdk is not on sys.path
    from abc import ABC, abstractmethod
    from dataclasses import dataclass

    class BaseModel(ABC):
        def __init__(self, model_path=None, output_path=None, seed=9899):
            self.model_path = model_path
            self.output_path = output_path
            self.seed = seed

    @dataclass
    class ModelOutput:
        predictions: object
        confidence: object = None
        metadata: object = None

        

class TimeMoeRunner(BaseModel):
    def __init__(
            self,
            model_path: str = None,
            output_path: str = 'logs/time_moe',
            seed: int = 9899
    ):
        super().__init__(model_path, output_path, seed)

        # Cache for parquet file to avoid loading it multiple times
        self._cached_df = None
        self._cached_df_path = None
        self.model = None

    def load_model(self, model_path: str = None, from_scatch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path
        attn = kwargs.pop('attn_implementation', None)
        input_size = kwargs.pop('input_size', None) ### Manually setting hte input size
        channel_configs = kwargs.pop('channel_configs', [])
        embedding_hidden_size = kwargs.pop('embedding_hidden_size', 128)
        if attn is None:
            attn = 'eager'
        elif attn == 'auto':
            # try to use flash-attention
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
                attn = 'flash_attention_2'
            except:
                log_in_local_rank_0('Flash attention import failed, switching to eager attention.', type='warn')
                attn = 'eager'

        if attn == 'eager':
            log_in_local_rank_0('Use Eager Attention')
        elif attn == 'flash_attention_2':
            log_in_local_rank_0('Use Flash Attention 2')
        else:
            raise ValueError(f'Unknown attention method: {attn}')
        kwargs['attn_implementation'] = attn

        if from_scatch:
            config = TimeMoeConfig.from_pretrained(model_path, _attn_implementation=attn)
            #log_in_local_rank_0(f'Config Input Size: {config.input_size}')
            
            # Adding in our own params
            config.input_size = input_size if input_size else config.input_size #### Manually setting input size
            config.channel_configs = channel_configs
            config.embedding_hidden_size = embedding_hidden_size
            
            # Add from the kwargs
            for key, value in kwargs.pop('model_configs', {}).items():
                setattr(config, key, value)
                assert getattr(config, key) == value, f'Set Model Param: {key} is different {getattr(config, key)} vs {value}'
            
            model = TimeMoeForPrediction(config)
        else:
            model = TimeMoeForPrediction.from_pretrained(model_path, **kwargs)
        return model

    def train_model(self, from_scratch: bool = False, optuna_search: bool = False, **kwargs):
        setup_seed(self.seed)
        args = kwargs.pop('args')
        train_config = kwargs
        num_devices = get_world_size()

        global_batch_size = train_config.get('global_batch_size', None)
        micro_batch_size = train_config.get('micro_batch_size', None)
        
        if global_batch_size is None and micro_batch_size is None:
            raise ValueError('Must set at lease one argument: "global_batch_size" or "micro_batch_size"')
        elif global_batch_size is None:
            gradient_accumulation_steps = 1
            global_batch_size = micro_batch_size * num_devices
        elif micro_batch_size is None:
            micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = 1
        else:
            if micro_batch_size * num_devices > global_batch_size:
                if num_devices > global_batch_size:
                    micro_batch_size = 1
                    global_batch_size = num_devices
                else:
                    micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = math.ceil(global_batch_size / num_devices / micro_batch_size)
            global_batch_size = int(gradient_accumulation_steps * num_devices * micro_batch_size)

        if ('train_steps' in train_config
                and train_config['train_steps'] is not None
                and train_config['train_steps'] > 0):
            train_steps = int(train_config["train_steps"])
            num_train_epochs = -1
        else:
            train_steps = -1
            num_train_epochs = _safe_float(train_config.get("num_train_epochs", 1))

        precision = train_config.get('precision', 'bf16')
        if precision not in ['bf16', 'fp16', 'fp32']:
            log_in_local_rank_0(f'Precision {precision} is not set, use fp32 default!', type='warn')
            precision = 'fp32'

        if precision == 'bf16':
            torch_dtype = torch.bfloat16
        elif precision == 'fp16':
            # use fp32 to load model but uses fp15 to train model
            torch_dtype = torch.float32
        elif precision == 'fp32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f'Unsupported precision {precision}')

        log_in_local_rank_0(f'Set global_batch_size to {global_batch_size}')
        log_in_local_rank_0(f'Set micro_batch_size to {micro_batch_size}')
        log_in_local_rank_0(f'Set gradient_accumulation_steps to {gradient_accumulation_steps}')
        log_in_local_rank_0(f'Set precision to {precision}')
        log_in_local_rank_0(f'Set normalization to {train_config["normalization_method"]}')

        model_path = train_config.pop('model_path', None) or self.model_path
        if model_path is not None:
            
            # Reading for optimal model_config parameters
            model_configs = train_config.get('model_configs', {})
            model = self.load_model(
                model_path=model_path,
                from_scatch=from_scratch,
                torch_dtype=torch_dtype,
                attn_implementation=train_config.get('attn_implementation', 'eager'),
                input_size=train_config.get('input_size'),
                channel_configs=train_config.get('channel_configs'),
                **model_configs
            )
            log_in_local_rank_0(str(model))
            log_in_local_rank_0(f'Load model parameters from: {model_path}')
        else:
            raise ValueError('Model path is None')

        num_total_params = 0
        for p in model.parameters():
            num_total_params += reduce(mul, p.shape)

        # print statistics info
        #log_in_local_rank_0(train_config)
        
        log_in_local_rank_0(model.config)
        log_in_local_rank_0(f'Number of the model parameters: {length_to_str(num_total_params)}')
        

        if train_steps > 0:
            total_train_tokens = train_steps * global_batch_size * train_config['max_length']
            log_in_local_rank_0(f'Tokens will consume: {length_to_str(total_train_tokens)}')
        
        if not optuna_search:
            training_args = self.load_training_argument(train_config, num_train_epochs, train_steps, micro_batch_size, gradient_accumulation_steps, precision)
            train_ds = self.get_dataset_multifrequency(args, 
                                                    'train', 
                                                    train_config['max_length'], 
                                                    train_config['stride'],
                                                    normalization_method=train_config["normalization_method"],
                                                    prediction_length=train_config.get('prediction_length', 0), 
                                                    load_prepared=train_config['load_prepared'], save_prepared=train_config['save_prepared'], 
                                                    prepared_save_path=train_config['prepared_save_path'])
            
            eval_ds = self.get_dataset_multifrequency(args, 
                                                    'test', 
                                                    train_config['max_length'], 
                                                    train_config['stride'],
                                                    normalization_method=train_config["normalization_method"],
                                                    prediction_length=train_config.get('prediction_length', 0), 
                                                    load_prepared=train_config['load_prepared'], save_prepared=train_config['save_prepared'], 
                                                    prepared_save_path=train_config['prepared_save_path'])
            
            log_in_local_rank_0(f'Initializaing Model Training full without Optuna Search...')
            log_in_local_rank_0(training_args)
            trainer = TimeMoeTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=collate_multi_frequency_fn,
                num_channels=len(train_config.get('channel_configs')) - 1, # Minus one to take away the main price channel
                callbacks=[EarlyStoppingCallback(early_stopping_patience=train_config.get('early_stopping_patience', 3),
                                                early_stopping_threshold=train_config.get('early_stopping_threshold', 0.0))]
            )
            
            # Get the checkpoint path
            checkpoint_path = train_config.get('resume_from_checkpoint')
            if checkpoint_path:
                log_in_local_rank_0(f'Resume training from checkpoint @ {checkpoint_path}')
                # Pass checkpoint to train() - the state will be loaded then
                trainer.train(resume_from_checkpoint=checkpoint_path)
                log_in_local_rank_0(f"Resumed from global step: {trainer.state.global_step}")
                log_in_local_rank_0(f"Resumed from epoch: {trainer.state.epoch}")
            else:
                trainer.train()
                
            trainer.save_model(self.output_path)
            log_in_local_rank_0(f'Saving model to {self.output_path}')

        else:
            # Optuna HyperParameter Search
            optuna_grid = kwargs.pop('optuna_grid', {})
            
            # Get hyperparameter search configuration
            n_trials = optuna_grid.pop('n_trials', 20)
            hp_train_size, hp_test_size = optuna_grid.get('optuna_hyper_param_dataset_sizes', (0.1, 0.1))
            
            if len(optuna_grid) == 0:   
                raise ValueError('Optuna search is enabled but the grid is empty!')
            
            log_in_local_rank_0(f'Initializaing Model Training with Optuna Search...')
            train_ds = self.get_dataset_multifrequency(args, 
                                                    'train', 
                                                    train_config['max_length'], 
                                                    train_config['stride'],
                                                    normalization_method=train_config["normalization_method"],
                                                    prediction_length=train_config.get('prediction_length', 0))
            
            eval_ds = self.get_dataset_multifrequency(args, 
                                                    'test', 
                                                    train_config['max_length'], 
                                                    train_config['stride'],
                                                    normalization_method=train_config["normalization_method"],
                                                    prediction_length=train_config.get('prediction_length', 0))
            
            # Base Training Args
            training_args = self.load_training_argument(
                train_config, 
                num_train_epochs, 
                train_steps, 
                micro_batch_size, 
                gradient_accumulation_steps, 
                precision
            )
            
            trainer = TimeMoeTrainer(
                model=None,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=collate_multi_frequency_fn,
                num_channels=len(train_config.get('channel_configs')) - 1, # Minus one to take away the main price channel
                model_init=lambda trial: self.model_init(
                    trial=trial,
                    model_path=model_path,
                    from_scratch=from_scratch,
                    torch_dtype=torch_dtype,
                    train_config=train_config,
                    model_grid=optuna_grid.get('model_grid').__dict__
                )
            )
            
            log_in_local_rank_0(f'Initializing Model Training with Optuna Search...')
            
            # Run hyperparameter search
            best_run = trainer.hyperparameter_search(
                direction="minimize",
                backend="optuna",
                hp_space=lambda trial: self.hp_space(trial, training_grid=optuna_grid.get('training_grid', {})),
                n_trials=n_trials,
                compute_objective=lambda metrics: self.compute_objective(metrics),
            )
            
            # Print best hyperparameters
            log_in_local_rank_0("\n" + "="*60)
            log_in_local_rank_0("Hyperparameter Search Complete!")
            log_in_local_rank_0("="*60)
            log_in_local_rank_0(f"Best run ID: {best_run.run_id}")
            log_in_local_rank_0(f"Best objective (eval_loss): {best_run.objective:.4f}")
            log_in_local_rank_0("\nBest hyperparameters:")
            for key, value in best_run.hyperparameters.items():
                log_in_local_rank_0(f"{key}: {value}")
                
            if optuna_grid.get('save_best_hyperparams'):
                best_params_path = optuna_grid.get('best_hyperparams_path')
                with open(best_params_path, 'w') as f:
                    json.dump(best_run.hyperparameters, f, indent=2)
                log_in_local_rank_0(f"Saved best hyperparameters to {best_params_path}")
            
            
        # Catch as model attribute
        self.model = trainer.model
        
        return trainer.model, best_run if optuna_search else None
    
    
    def model_init(
        self, trial, model_path: str = None, from_scratch: bool = False, 
        torch_dtype=torch.float32, train_config: dict = None, model_grid: dict = None,
    ):
        """
        Initialize a TimeMoE model for Optuna hyperparameter tuning or single-run training.
        """
        train_config = train_config or {}
        model_grid = model_grid or {}
        # ---- Load base config ----
        if model_path:
            config = TimeMoeConfig.from_pretrained(
                model_path,
                _attn_implementation=train_config.get("attn_implementation", "eager"),
            )
            log_in_local_rank_0(f"Loaded base config from {model_path}")
        else:
            config = TimeMoeConfig()
            log_in_local_rank_0("Initialized new base config from default settings")

        # ---- Apply base train_config overrides ----
        if "input_size" in train_config:
            config.input_size = train_config["input_size"]
        if "channel_configs" in train_config:
            config.channel_configs = train_config["channel_configs"]
        if "embedding_hidden_size" in train_config:
            config.embedding_hidden_size = train_config["embedding_hidden_size"]

        # ---- Handle hyperparameters from Optuna ----
        if trial is not None and model_grid:
            # Integers (but defined as categorical discrete values)
            if "int" in model_grid:
                for name, choices in model_grid["int"].__dict__.items():
                    if isinstance(choices, (list, tuple)):
                        value = trial.suggest_categorical(f"model_{name}", list(choices))
                        setattr(config, name, int(value))
                        #log_in_local_rank_0(f"Trial sets {name} = {value}")
                    else:
                        log_in_local_rank_0(f"Invalid int grid for {name}: {choices}", type="warn")

            # Floats
            if "float" in model_grid:
                for name, rng in model_grid["float"].__dict__.items():
                    if isinstance(rng, (list, tuple)) and len(rng) == 2:
                        low, high = float(rng[0]), float(rng[1])
                        value = trial.suggest_float(f"model_{name}", low, high)
                        setattr(config, name, value)
                        #log_in_local_rank_0(f"Trial sets {name} = {value}")
                    else:
                        log_in_local_rank_0(f"Invalid float range for {name}: {rng}", type="warn")

            # Categoricals
            if "categorical" in model_grid:
                for name, choices in model_grid["categorical"].__dict__.items():
                    if isinstance(choices, (list, tuple)):
                        value = trial.suggest_categorical(f"model_{name}", list(choices))
                        setattr(config, name, value)
                        #log_in_local_rank_0(f"Trial sets {name} = {value}")
                    else:
                        log_in_local_rank_0(f"Invalid categorical grid for {name}: {choices}", type="warn")

        else:
            log_in_local_rank_0("No Optuna trial — using base config only.")

        # ---- Final config summary ----
        log_in_local_rank_0(f"Final model config: {config}")

        # ---- Initialize model ----
        model = TimeMoeForPrediction(config)

        log_in_local_rank_0(str(model))
        log_in_local_rank_0(str(config))
        return model
    
    def hp_space(self, trial, training_grid: dict = None):
        '''
        Updates hyper parameter search space based on the grid
        '''
        if training_grid is None:
            training_grid = {}
            
        res = {}
        
        for param_type, params_dict in training_grid.__dict__.items():
            for param_name, param_args in params_dict.__dict__.items():
                
                # Handle each parameter type separately
                if param_type == 'categorical':
                    # Categorical: pass the list directly
                    res[param_name] = trial.suggest_categorical(param_name, param_args)
                    
                elif param_type == 'float':
                    # Float: unpack min/max, use log scale for learning_rate
                    if param_name == 'learning_rate':
                        res[param_name] = trial.suggest_float(param_name, *param_args, log=True)
                    else:
                        res[param_name] = trial.suggest_float(param_name, *param_args)
                        
                elif param_type == 'int':
                    # Int: unpack min/max
                    res[param_name] = trial.suggest_int(param_name, *param_args)
                    
                else:
                    raise ValueError(f'Unknown hyperparameter search type: {param_type}. Must be one of ["float", "int", "categorical"]')
        
        return res
    
    
    def compute_objective(self, metrics):
        """Compute the objective value for Optuna optimization"""
        return metrics['eval_loss']

    
    def load_training_argument(self, train_config, num_train_epochs, train_steps, micro_batch_size, gradient_accumulation_steps, precision):
        '''
        Wrapper that returns the TimeMoETrainingArguments given the training config
        '''
        training_args = TimeMoETrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=num_train_epochs,
            # use_cpu=True,
            max_steps=train_steps,
            evaluation_strategy=train_config.get("evaluation_strategy", 'no'),
            eval_steps=_safe_float(train_config.get("eval_steps", None)),
            save_strategy=train_config.get("save_strategy", "no"),
            save_steps=_safe_float(train_config.get("save_steps", None)),
            learning_rate=float(train_config.get("learning_rate", 1e-5)),
            min_learning_rate=float(train_config.get("min_learning_rate", 0)),
            adam_beta1=float(train_config.get("adam_beta1", 0.9)),
            adam_beta2=float(train_config.get("adam_beta2", 0.95)),
            adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
            lr_scheduler_type=train_config.get("lr_scheduler_type", 'constant'),
            warmup_ratio=float(train_config.get("warmup_ratio") or 0.0),
            warmup_steps=int(train_config.get("warmup_steps", 0)),
            weight_decay=float(train_config.get("weight_decay", 0.1)),
            per_device_train_batch_size=int(micro_batch_size),
            per_device_eval_batch_size=int(micro_batch_size * 2),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            gradient_checkpointing=train_config.get("gradient_checkpointing", False),
            bf16=True if precision == 'bf16' else False,
            fp16=True if precision == 'fp16' else False,
            deepspeed=train_config.get("deepspeed"),
            push_to_hub=train_config.get(f'push_to_hub', False),
            hub_token=os.environ.get('HF_TOKEN'),
            logging_first_step=True,
            log_on_each_node=False,
            logging_steps=int(train_config.get('logging_steps', 50)),
            seed=self.seed,
            data_seed=self.seed,
            max_grad_norm=train_config.get('max_grad_norm', 1.0),
            optim=train_config.get('optim', 'adamw_torch'),
            
            torch_compile=train_config.get('torch_compile', False),
            greater_is_better=train_config.get('greater_is_better', False),
            
            dataloader_num_workers=train_config.get('dataloader_num_workers', 0),
            dataloader_persistent_workers=True if train_config.get('dataloader_num_workers', 0) > 0 else False,
            dataloader_pin_memory=train_config.get('dataloader_pin_memory', True),
            ddp_find_unused_parameters=False,
            logging_dir=os.path.join(self.output_path, 'tb_logs'),
            save_only_model=train_config.get('save_only_model', True),
            save_total_limit=train_config.get('save_total_limit'),
            
            # Early Stopping
            load_best_model_at_end=train_config.get('load_best_model_at_end', False),
            metric_for_best_model=train_config.get('metric_for_best_model', 'eval/loss'),
            
            # Resume checkpoint
            resume_from_checkpoint=train_config.get('resume_from_checkpoint', None)
        )
        return training_args
    

    def get_train_dataset(self, data_path, max_length, stride, normalization_method, prediction_length=0):
        log_in_local_rank_0(f'Loading dataset... @ {data_path}')
        dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
        log_in_local_rank_0(f'Loaded dataset of cumsum length: {dataset.num_sequences}. Total number of datasets: {len(dataset.datasets)}...')
        log_in_local_rank_0('Processing dataset to fixed-size sub-sequences....')
        window_dataset = TimeMoEWindowDataset(dataset, context_length=max_length, prediction_length=prediction_length, stride=stride, shuffle=False)
        log_in_local_rank_0(f'Total Dataset Size: {len(window_dataset)}....')

        return window_dataset
    
    
    def get_dataset_multifrequency(self, args, stage, # Path to the feature engineered dataframe
                                         max_length,
                                         stride,
                                         normalization_method, prediction_length=0,
                                         load_prepared=False, save_prepared=False, prepared_save_path=None):
        '''
        Read in from data_path in args and then use the MultiFrequencDatasetLoader
        '''
        data_path = args.regression.S2.feature_engineered_file_path
        resample_params = []
        # Use cached dataframe to avoid loading parquet multiple times
        if self._cached_df is None or self._cached_df_path != data_path:
            log_in_local_rank_0(f'Loading dataset from disk... @ {data_path}')
            self._cached_df = pd.read_parquet(data_path)
            self._cached_df_path = data_path
        else:
            log_in_local_rank_0(f'Using cached dataset (avoiding disk I/O)')

        full_df = self._cached_df
        base_dataset = MultiFrequencyTimeSeriesDataset(args, full_df, stage)
        log_in_local_rank_0('Processing dataset to fixed-size sub-sequences....')
        
        if args.regression.S3.resample is True and stage == 'train':
            # We get out all the resampling parameters
            dict_form = args.regression.S3.bucket_mapping_jsons.__dict__
            for k, v in dict_form.items():
                resample_params.append({'buckets': v.__dict__['buckets'], 'weights': v.__dict__['weights']})

        window_dataset = TimeMoEMultiFrequencyWindowDatasetV2(
                            base_dataset, 
                            max_length, 
                            prediction_length,
                            stride,
                            normalization_method=normalization_method,
                            inferencing_mode=True if stage == 'inference' else False,
                            resample_weights_path=resample_params
                            #load_prepared=load_prepared, save_prepared=save_prepared, save_path=prepared_save_path,
                        )
        
        
        log_in_local_rank_0(f'Total Dataset Size: {len(window_dataset)}....')
        return window_dataset
    
    
    
    def predict(self, input_batch: dict = None, model=None, device=None, model_path: str = None,
                max_horizon_length: int = 1):
        '''
        Model inferencing function.

        Args:
            input_batch:         Dict with at minimum 'input_ids' tensor and optional
                                 'channel_*' tensors plus 'mean'/'std' for inverse scaling.
            model:               Optional pre-loaded model. Uses self.model if not provided.
            device:              Device override (e.g. 'cuda:0'). Auto-detected if None.
            model_path:          HF Hub ID or local path to load model from if no model is set.
            max_horizon_length:  Number of forecast steps to generate.
        '''
        if not input_batch:
            log_in_local_rank_0('No inputs detected! input_batch is None!')
            raise ValueError('No inputs detected')

        if model is None and self.model is None:
            selected_path = self.model_path if model_path is None else model_path
            log_in_local_rank_0(f"Model set as None! Loading from HuggingFace @ {selected_path}")
            self.model = TimeMoeForPrediction.from_pretrained(
                selected_path,
                device_map='cpu' if not torch.cuda.is_available() else ('auto' if device is None else device),
                torch_dtype='auto',
            )

        active_model = model if model is not None else self.model

        with torch.no_grad():
            if torch.isnan(input_batch['input_ids']).any():
                print("NaN detected in input subsequence_norm!")

            output = active_model(
                input_ids=input_batch['input_ids'],
                max_horizon_length=max_horizon_length,
                return_dict=True,
                **{key: input_batch[key] for key in input_batch if key.startswith('channel_')}
            )
            
            #print(f'Output Shape: {output.logits.shape}')
            forecast = output.logits[:, -1, :]   # [B, prediction_len, F] simplified

            #print(f'Forecast Shape: {forecast.shape}')
            
            # --- Inverse scaling per subsequence ---
            forecast_original_scale = []
            for i in range(forecast.shape[0]):
                f = forecast[i]                         # [prediction_len, F]
                mean = input_batch['mean'][i]           # [1, F]
                std = input_batch['std'][i]             # [1, F]
                f_inv = inverse_scale(f, mean, std)
                forecast_original_scale.append(f_inv)

            forecast_original_scale = torch.stack(forecast_original_scale, dim=0)  # [B, prediction_len, F]

            # Store in batch
            input_batch['model_prediction_sequence'] = forecast_original_scale
        
        return ModelOutput(predictions=input_batch)

def setup_seed(seed: int = 9899):
    """
    Setup seed for all known operations.

    Args:
        seed (int): seed number.

    Returns:

    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def length_to_str(length):
    if length >= 1e12:
        return f'{length / 1e12:.3f}T'
    if length >= 1e9:
        return f'{length / 1e9:.3f}B'
    elif length >= 1e6:
        return f'{length / 1e6:.3f}M'
    else:
        return f'{length / 1e3:.3f}K'


def _safe_float(number):
    if number is None:
        return None
    else:
        return float(number)


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
