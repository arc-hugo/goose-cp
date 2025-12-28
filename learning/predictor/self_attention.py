import logging
import math

import torch
import comet_ml

from collections import OrderedDict

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base_predictor import BaseEpochPredictor


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def linear_sequence(in_dim: int, out_dim: int, hidden_dim: int,
                    num_hidden: int, name_block: str):
    
    sequence = [(f"{name_block}_linear1", nn.Linear(in_dim, in_dim)), (f"{name_block}_relu1", nn.ReLU())]
    if num_hidden > 0:
        sequence += [(f"{name_block}_linear2", nn.Linear(in_dim, hidden_dim)), (f"{name_block}_relu2", nn.ReLU())]
        for i in range(3,num_hidden+2):
            sequence += [(f"{name_block}_linear{i}", nn.Linear(hidden_dim, hidden_dim)), (f"{name_block}_relu{i}", nn.ReLU())]
        sequence += [(f"{name_block}_linear{num_hidden+2}", nn.Linear(hidden_dim, out_dim))]
    else :
        sequence += [(f"{name_block}_linear2", nn.Linear(in_dim, out_dim))]

    return sequence

class SASoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_hidden=1, dropout=0.25):
        super(SASoftmaxModel, self).__init__()

        self._hidden_dim = hidden_dim
        self._num_hidden = num_hidden
        
        # Query encoder block
        self.Q_fc = nn.Sequential(OrderedDict(linear_sequence(
            input_dim, hidden_dim, hidden_dim, num_hidden, "Q"
        )))

        # Key encoder block
        self.K_fc = nn.Sequential(OrderedDict(linear_sequence(
            input_dim, hidden_dim, hidden_dim, num_hidden, "K"
        )))

        # Value encoder block
        self.V_fc = nn.Sequential(OrderedDict(linear_sequence(
            input_dim, hidden_dim, hidden_dim, num_hidden, "V"
        )))

        # Multihead block
        self.multihead = nn.MultiheadAttention(
            hidden_dim, 1, dropout, batch_first=True
        )

        # Encoder block
        # self.input_fc = nn.Sequential(OrderedDict(linear_sequence(
        #     input_dim, hidden_dim, hidden_dim, num_hidden, "input"
        # )))

        # Decoder block
        self.output_fc = nn.Sequential(OrderedDict(linear_sequence(
            hidden_dim, 1, hidden_dim, num_hidden, "output"
        )))


        # Apply He init to all blocks
        self.Q_fc.apply(weights_init)
        self.V_fc.apply(weights_init)
        self.K_fc.apply(weights_init)
        # self.input_fc.apply(weights_init)
        self.output_fc.apply(weights_init)

        # Compute scale factor
        self.scale_factor = 1 / math.sqrt(hidden_dim)
        
    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_length, input_dim)

        # x = x.unsqueeze(1) # Shape: (batch_size, num_head, seq_length, input_dim)

        # Apply blocks Query, Key, Value
        Q = self.Q_fc(x)
        K = self.K_fc(x)
        V = self.V_fc(x)
        
        # inp = self.input_fc(x)

        # Compute scaled dot product attention
        dot_prod, _ = self.multihead(Q, K, V)

        # Apply output block on dot product
        out = self.output_fc(dot_prod)

        # Supprimer de la deuxième et la dernière dimension
        out = out.squeeze(-1) #.squeeze(1) # Shape: (batch_size, seq_length)
        
        # Appliquer softmax sur la dimension de la séquence
        out = F.log_softmax(out, dim=1)
        
        return out

class SelfAttentionSoftmax(BaseEpochPredictor):
    def __init__(self, input_dim: int, domain: str, action_schema: str, iterations: int,
                 criterion=nn.KLDivLoss, optimizer=AdamW,
                 epoch=1000, alpha=1e-3, device="cuda:0"):
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = SASoftmaxModel(input_dim)
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

        super().__init__(domain, action_schema, iterations, epoch=epoch, alpha=alpha, log_params=opt_params)

    def _train_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment) -> float:
        self._model.train()
        total_loss = 0
        nb_data = 0
        first = True

        for train_data in data:
            for X,y in train_data:
                # Pass data to device
                X, y = X.to(self._device), y.to(self._device)

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
                self.optimizer.zero_grad()

                total_loss += loss.item()
                nb_data += 1
        
        total_loss /= nb_data
        exp.log_metrics({"loss": total_loss}, epoch=epoch)

        logging.info(f"Avg loss: {total_loss:>8f}")
        self._fitted = True

        return total_loss
    
    def _validate_impl(self, data: DataLoader, epoch: int, exp: comet_ml.CometExperiment) -> float:
        self._model.eval()

        total_loss, num_batches = 0, 0
    
        with torch.no_grad():
            for train_data in data:
                for X,y in train_data:
                    # Pass data to device
                    X, y = X.to(self._device), y.to(self._device)
                    
                    pred = self._model(X)

                    total_loss += self.criterion(pred, y).item()
                    num_batches += 1

        total_loss /= num_batches

        exp.log_metrics({"loss": total_loss}, epoch=epoch)
        logging.info(f"Avg validation loss: {total_loss:>8f}")

        return total_loss

    def get_model(self) -> nn.Module:
        return self._model
    

        

