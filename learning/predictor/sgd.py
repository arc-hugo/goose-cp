import logging
import numpy as np
import torch
import comet_ml

from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from collections import OrderedDict

from .base_predictor import BaseCPPredictor

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

class LinearSoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_hidden=4):
        super(LinearSoftmaxModel, self).__init__()

        self._hidden_dim = hidden_dim
        self._num_hidden = num_hidden
        
        sequence = [("linear1", nn.Linear(input_dim, input_dim, dtype=torch.float64)), ("relu1", nn.ReLU())]
        if num_hidden > 0:
            sequence += [("linear2", nn.Linear(input_dim, hidden_dim, dtype=torch.float64)), ("relu2", nn.ReLU())]
            for i in range(3,num_hidden+2):
                sequence += [("linear"+str(i), nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64)), ("relu"+str(i), nn.ReLU())]
            sequence += [("linear"+str(num_hidden+2), nn.Linear(hidden_dim, 1, dtype=torch.float64))]
        else :
            sequence += [("linear2", nn.Linear(input_dim, 1, dtype=torch.float64))]

        self.seq = nn.Sequential(OrderedDict(sequence))

        # Apply He init
        self.seq.apply(weights_init)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # Appliquer les couches linéaires
        x = self.seq(x)
        
        # Supprimer la dernière dimension
        x = x.squeeze(-1)  # Shape: (batch_size, seq_length)
        
        # Appliquer softmax sur la dimension de la séquence
        x = F.log_softmax(x, dim=1)
        
        return x

class RegressorSoftmax(BaseCPPredictor):
    def __init__(self, input_dim: int, domain: str, action_schema: str, iterations: int,
                 criterion=nn.KLDivLoss, optimizer=torch.optim.Adam, epoch=1000, alpha=1e-2,
                 device="cuda:0"):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = LinearSoftmaxModel(input_dim)
        self._model.to(self._device)

        self._fitted = False

        self.criterion = criterion(reduction='batchmean')
        self.optimizer = optimizer(self._model.parameters(), alpha)

        opt_params = {
            "criterion": self.criterion.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "hidden_dim": self._model._hidden_dim,
            "num_hidden": self._model._num_hidden
        }

        super().__init__(domain, action_schema, iterations, epoch=epoch, alpha=alpha, opt_params=opt_params)

    def _train_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment):
        self._model.train()
        total_loss = 0
        nb_data = 0
        first = True

        for _, (X, y) in enumerate(data):
            # Pass data to device
            X, y = X.to(self._device), y.to(self._device)
            
            self.optimizer.zero_grad()

            # Compute prediction and loss
            pred = self._model(X)

            if first:
                print(pred)
                print(y)
                first = False

            loss = self.criterion(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            nb_data += 1
        
        total_loss /= nb_data
        exp.log_metrics({"loss": total_loss}, epoch=epoch)

        logging.info(f"Avg loss: {total_loss:>8f}")
        self._fitted = True
    
    def _evaluate_impl(self, data: DataLoader):
        # self._model.eval()

        # test_loss, num_batches = 0, 0
    
        # with torch.no_grad():
        #     for X, y in data:
        #         # Pass data to device
        #         X, y = X.to(self._device), y.to(self._device)
                
        #         pred = self._model(X)

        #         test_loss += self.criterion(pred, y).item()
        #         num_batches += 1

        # test_loss /= num_batches
        # logging.info(f"Avg loss: {test_loss:>8f}")
        pass

    def get_model(self):
        return self._model

    def _save_weights(self):
        self._weights = self._model.state_dict()["linear1.weight"].squeeze().tolist()

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model has not been trained yet. Call `fit` to train the model.")

        self._model.eval()
        return self._model.forward(X)
    
    def set_weights(self, weights):
        self._weights = np.array(weights)
        self._model.coef_ = np.array(weights)
        self._model.intercept_ = np.zeros((1,))
        self._fitted = True
        

