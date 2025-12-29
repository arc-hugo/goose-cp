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
    def __init__(self, input_dim, hidden_dim=1024, num_head=4, dropout=0.1, qkv_bias=False):
        super(SASoftmaxModel, self).__init__()
        assert hidden_dim % num_head == 0
        
        self._hidden_dim = hidden_dim
        self._num_head = num_head
        self._head_dim = hidden_dim // num_head
        
        # Encoder block for queries, keys and values
        self.QKV_fc = nn.Linear(input_dim, 3 * hidden_dim, bias=qkv_bias)

        # Decoder block
        self.output_fc = nn.Linear(hidden_dim, 1)

        # Dropout block
        self.dropout = dropout

        # Apply He init to all blocks
        self.QKV_fc.apply(weights_init)
        # self.input_fc.apply(weights_init)
        self.output_fc.apply(weights_init)

        # Compute scale factor
        self.scale_factor = 1 / math.sqrt(hidden_dim)
        
    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_size, input_dim)
        batch_size, seq_size, _ = x.shape

        # x = x.unsqueeze(1) # Shape: (batch_size, num_head, seq_size, input_dim)

        # Apply blocks Query, Key, Value
        qkv: torch.Tensor = self.QKV_fc(x)

        qkv = qkv.view(batch_size, seq_size, 3, self._num_head, self._head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv
        
        # Dropout
        dropout = 0. if not self.training else self.dropout

        # Compute scaled dot product attention
        context_vec = nn.functional.scaled_dot_product_attention(
             Q, K, V, attn_mask=None, dropout_p=dropout
        )
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, seq_size, self._hidden_dim)

        # Output projection
        out = self.output_fc(context_vec)

        # Supprimer de la deuxième et la dernière dimension
        out = out.squeeze(-1) #.squeeze(1) # Shape: (batch_size, seq_size)
        
        # Appliquer softmax sur la dimension de la séquence
        out = F.log_softmax(out, dim=1)
        
        return out

class SelfAttentionSoftmax(BaseEpochPredictor):
    def __init__(self, input_dim: int, domain: str, action_schema: str, iterations: int,
                 criterion=nn.KLDivLoss, optimizer=AdamW,
                 epoch=3000, alpha=1e-4, device="cuda:0"):
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
            "num_head": self._model._num_head
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
    

        

