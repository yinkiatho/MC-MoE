import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from argparse import Namespace
from sklearn.metrics import mean_squared_error, accuracy_score, root_mean_squared_error
import optuna  # for Bayesian optimization
from itertools import product
import os
from utils.utils import TensorKeyDataset

class ModelTrainer:
    def __init__(self, model: nn.Module, args: Namespace, device=None):
        """
        Trainer class for PyTorch models.

        Args:
            model: PyTorch nn.Module
            args: Namespace containing training args
            device: torch.device, defaults to cuda if available
        """
        self.model = model
        self.args = args
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _prepare_dataloader(self, all_tensors, shuffle=False, batch_size=32):
        """
        Convert list of (X_tensor, y_tensor) to a DataLoader
        """
        # X_all = torch.cat([x for x, y, unique_keys in all_tensors], dim=0)
        # y_all = torch.cat([y for x, y, unique_keys in all_tensors], dim=0)
        dataset = TensorKeyDataset(all_tensors)   
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _load_weights(self, model_weights_path: str):
        """
        Load model weights from a given file path.

        Args:
            model_weights_path (str): Path to the saved model weights (.pth file)
        """
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

        state_dict = torch.load(model_weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logging.info(f"Loaded model weights from {model_weights_path}")
    

    def train_loop(self, train_data, val_data=None, epochs=50, lr=1e-3, loss_fn=None, task='regression'):
        """
        Training loop with step-level loss and gradient norm tracking
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if not loss_fn:
            loss_fn = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()

        train_losses, val_losses = [], []
        step_train_losses, step_val_losses, step_grad_norms = [], [], []  # <-- step-level tracking

        step = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch, _ in train_data:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(X_batch)
                if task == 'classification' and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)

                loss = loss_fn(outputs, y_batch)
                loss.backward()

                # ---- compute grad norm per step ----
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)  # L2 norm of gradient
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                optimizer.step()

                # store step-level values
                step_train_losses.append(loss.item())
                step_grad_norms.append(total_norm)

                # update epoch stats
                epoch_loss += loss.item() * X_batch.size(0)
                step += 1

            epoch_loss /= len(train_data.dataset)
            train_losses.append(epoch_loss)

            # Validation
            if val_data:
                self.model.eval()
                val_loss = 0.0
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for X_val, y_val, _ in val_data:
                        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                        outputs = self.model(X_val)
                        if task == 'classification' and outputs.shape[1] == 1:
                            outputs = outputs.squeeze(1)
                        loss = loss_fn(outputs, y_val)
                        val_loss_step = loss.item() * X_val.size(0)
                        step_val_losses.append(val_loss_step)
                        val_loss += val_loss_step
                        all_preds.append(outputs.cpu())
                        all_targets.append(y_val.cpu())
                val_loss /= len(val_data.dataset)
                val_losses.append(val_loss)

                # Compute metric
                if task == 'regression':
                    metric = mean_squared_error(
                        torch.cat(all_targets).numpy(),
                        torch.cat(all_preds).numpy()
                    )
                    logging.info(
                        f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | MSE: {metric:.4f}"
                    )
                else:
                    preds = torch.cat(all_preds).argmax(dim=1)
                    metric = accuracy_score(torch.cat(all_targets).numpy(), preds.numpy())
                    logging.info(
                        f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Accuracy: {metric:.4f}"
                    )
            else:
                logging.info(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f}"
                )

        # return both epoch- and step-level traces
        return train_losses, val_losses, step_train_losses, step_val_losses, step_grad_norms

    def grid_search(self, all_tensors, val_tensors=None, param_grid=None, save_weights=True):
        """
        Run simple grid search over hyperparameters (lr, batch_size, epochs)
        param_grid example: {'lr':[1e-3,1e-4], 'batch_size':[32,64], 'epochs':[50,100]}
        """
        logging.info(f"Initializaing Grid Search Optimization for Training Tensor Size: {len(all_tensors)}, Val Tensors: {len(val_tensors)}")
        best_score = float('inf')
        best_params = None
        best_model_state = None
        best_train_log = {}
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            
            logging.info(f"Grid Search Optimization: trying params: {param_dict}")
            
            # Preparing Parameters
            train_loader = self._prepare_dataloader(all_tensors, batch_size=param_dict.get('batch_size'))
            val_loader = self._prepare_dataloader(val_tensors, batch_size=param_dict.get('batch_size')) if val_tensors else None
            self.model.apply(self._weights_init)  # Reset model weights

            # Main Train Loop
            train_losses, val_losses, step_train_losses, step_val_losses, step_grad_norms = self.train_loop(train_loader, val_loader, 
                                                                                                            lr=param_dict.get('lr'), 
                                                                                                            epochs=param_dict.get('epochs'),
                                                                                                            task='regression')
            if val_loader:
                score = val_losses[-1]
            else:
                score = train_losses[-1]

            if score < best_score:
                best_score = score
                best_params = param_dict
                best_model_state = self.model.state_dict()
                best_train_log = {'Train Losses': train_losses, 'Val Losses': val_losses, 'Step Train Losses': step_train_losses, 
                                  'Step Val Losses': step_val_losses, 'Step Gradient Norms': step_grad_norms}
        
        self.model.load_state_dict(best_model_state)
        logging.info(f"Best grid search params: {best_params} with score {best_score:.4f}")
        
        if save_weights:
            filename = save_weights if isinstance(save_weights, str) else f"models/model_weights_states/{self.args.regression.common.version_name}_model_weights.pth"
            torch.save(self.model.state_dict(), filename)
            logging.info(f"Grid Search Model weights saved to {filename}")
            return best_params, best_score, filename, best_train_log
        return best_params, best_score, '', best_train_log
        

    def bayesian_optimization(self, all_tensors, val_tensors=None, n_trials=20, search_space=None):
        """
        Optimize hyperparameters using Bayesian optimization (Optuna)
        search_space: a function defining hyperparameter search space
        """
        def objective(trial):
            lr = trial.suggest_float('lr', search_space['lr'][0], search_space['lr'][1], log=True)
            batch_size = trial.suggest_int('batch_size', search_space['batch_size'][0], search_space['batch_size'][1])
            epochs = trial.suggest_int('epochs', search_space['epochs'][0], search_space['epochs'][1])

            #self.args.batch_size = batch_size
            train_loader = self._prepare_dataloader(all_tensors, batch_size=batch_size)
            val_loader = self._prepare_dataloader(val_tensors, batch_size=batch_size) if val_tensors else None
            self.model.apply(self._weights_init)  # Reset weights
            _, val_losses = self.train_loop(train_loader, val_loader, epochs=epochs, lr=lr, task=self.args.task)
            return val_losses[-1]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logging.info(f"Best trial: {study.best_trial.params} with score {study.best_trial.value:.4f}")
        return study.best_trial.params, study.best_trial.value

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def run_inferencing(self, model_weights_path: str, inference_data: list, batch_size: int = 32):
        '''
        Function runs the inferencing using the train/model weights initialised model
        '''
        logging.info(f"Initiating S3.2 Model Inferencing on total of {len(inference_data)} tensors")
        # Initialise the model weights
        self._load_weights(model_weights_path=model_weights_path)
        
        # Prepare data loader for inference
        inference_loader = self._prepare_dataloader(inference_data, shuffle=False, batch_size=batch_size)
        
        # Run inferencing on the tensors
        self.model.eval()
        all_predictions, all_targets, all_inputs, all_keys = [], [], [], []
        
        with torch.no_grad():
            for X_batch, y_batch, keys_batch in inference_loader:
                #print(f"Inferencning for keys: {keys_batch}")
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Handle classification case where output might need squeezing
                task = getattr(self.args, 'task', 'regression')
                if task == 'classification' and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)

                all_predictions.append(outputs.cpu())
                all_targets.append(y_batch.cpu())
                all_inputs.append(X_batch.cpu())
                all_keys.extend(keys_batch)   
        
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)        
        inputs = torch.cat(all_inputs, dim=0).numpy()
        
        results = {
            'inputs': inputs,
            'predictions': predictions.numpy(),
            'targets': targets.numpy(),
            'keys': all_keys, 
            'predictions_tensor': predictions,
            'targets_tensor': targets
        }
        
        logging.info(f"Inference completed.................")
        return results
            