import logging

import numpy as np

from sklearn.metrics import f1_score, mean_squared_error
from scipy.special import softmax

from .base_predictor import BasePredictor, BaseCPPredictor


class MSEMinimiser(BasePredictor):
    IS_RANK = False

    def _evaluate_impl(self):
        X, y = self._X, self._y
        y_pred = self.predict(X)
        mse_loss = mean_squared_error(y, y_pred, sample_weight=self._sample_weight)
        logging.info(f"{mse_loss=}")
        y_pred = np.round(y_pred)
        f1_macro = f1_score(y, y_pred, average="macro", sample_weight=self._sample_weight)
        logging.info(f"{f1_macro=}")

class MSESoftmaxMinimiser(BaseCPPredictor):
    def _evaluate_impl(self):
        X_train = self._X
        
        y = np.array(self._y).flatten()
        y_softmax = []
        first = True

        for X in X_train:
            y_pred = self.predict(X)
            y_softmax = np.concatenate([y_softmax, softmax(y_pred)])
            if first:
                print(self._y[0])
                print(y_softmax)
                first = False

        mse_loss = mean_squared_error(y, y_softmax)
        logging.info(f"{mse_loss=}")
        y_softmax = np.round(y_softmax)
        f1_macro = f1_score(y, y_softmax, average="macro")
        logging.info(f"{f1_macro=}")