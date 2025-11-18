import math

from abc import ABC, abstractmethod

import comet_ml
from comet_ml.integration.pytorch import log_model

import numpy as np

from torch.utils.data import DataLoader

from util.custom_decorators import abstract_class_attributes
from util.timer import TimerContextManager


@abstract_class_attributes("IS_RANK")
class BasePredictor(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._weights = None

    def fit_evaluate(self, X, y, sample_weight):
        with TimerContextManager("training model"):
            self.fit(X, y, sample_weight)
        with TimerContextManager("evaluating model"):
            self.evaluate()

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self._fit_impl(X, y, sample_weight)

    def evaluate(self):
        if self._weights is None:
            raise RuntimeError("Model has not been trained yet. Call `fit` to train the model.")
        self._evaluate_impl()

    @abstractmethod
    def _fit_impl(self, X, y, sample_weight):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _evaluate_impl(self):
        """Evaluation of training data after calling fit"""
        pass

    def get_weights(self) -> list:
        if self._weights is None:
            raise RuntimeError("Model has not been trained yet. Call `fit` to train the model.")
        ret = self._weights
        if isinstance(ret, np.ndarray):
            ret = ret.tolist()
        return ret

class BaseEpochPredictor(ABC):
    def __init__(self, domain: str, action_schema: str, iterations: int, epoch = 10, alpha=1e-3, log_params={}) -> None:
        super().__init__()

        self._weights = None
        self.epochs = epoch

        self.params = {
            "learning_rate": alpha,
            "domain": domain,
            "action_schema": action_schema,
            "iterations": iterations,
            "epochs": "inf",
            **log_params
        }

    def fit(self, train_data: DataLoader, validation_data: DataLoader):
        exp = comet_ml.start(
            project_name="goose-cp"
        )
        exp.disable_mp()
        exp.log_parameters(self.params)

        min_delta = 1e-5
        prev_loss = math.inf
        counter = 0
        
        t = 0
        while True:
            print(f"Epoch {t+1}\n-------------------------------")
            with exp.train():
                train_loss = self._train_impl(train_data, t, exp)

            # with exp.test():
            #     self._validate_impl(validation_data, t, exp)

            if (abs(prev_loss - train_loss) < min_delta):
                counter += 1
                if (counter > 10):
                    break
            else:
                counter = 0
            
            t += 1
        
        log_model(exp, self.get_model(), "LinearSoftmaxModel-CP")

        exp.end()
        
        return self

    @abstractmethod
    def _train_impl(self, data, epoch: int, exp: comet_ml.CometExperiment) -> float:
        """Train the model on a dataset object"""
        pass

    @abstractmethod
    def _validate_impl(self, data, epoch: int, exp: comet_ml.CometExperiment) -> float:
        """Process validation data alongside the training process"""
        pass
