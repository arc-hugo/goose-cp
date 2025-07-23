import numpy as np

from sklearn.linear_model import SGDRegressor as SGDR
from scipy.special import softmax

from .base_predictor import BaseCPPredictor

class SGDRegressorSoftmax(BaseCPPredictor):
    def __init__(self, alpha=0.0001):
        super().__init__()
        self._model = SGDR(alpha=alpha, tol=1e-7, fit_intercept=False)
        self.alpha = alpha
        self._fitted = False

    def _partial_fit_impl(self, X, y):
        if self._fitted:
            y_pred = self._model.predict(X)
        else:
            y_pred = np.random.randn(len(X))
        
        # Softmax transform
        y_softmax = softmax(y_pred)

        # MSE gradient
        error = y_softmax - y
        jacobian = np.diag(y_softmax) - np.outer(y_softmax, y_softmax)
        gradient = 2 * jacobian @ error

        # Compute new target
        y_target = y_pred - self.alpha * gradient

        # if not self._fitted:
        #     mse_loss = mean_squared_error(y, y_softmax)
        #     print(y)
        #     print(y_softmax)
        #     print(np.sum(y_softmax))
        #     logging.info(f"{mse_loss=}")
        # exit()
        # Apply partial fit
        self._model.partial_fit(X, y_target)

        # Update weights and train data
        self._weights = self._model.coef_
        self._fitted = True
    
    def predict(self, X, groups=None):
        if not self._fitted:
            raise RuntimeError("Model has not been trained yet. Call `partial_fit` to train the model.")
        if groups is None:
            groups = np.zeros(len(X))
        
        raw_pred = self._model.predict(X)

        return self._apply_group_softmax(raw_pred, groups)

    def _apply_group_softmax(self, raw_pred, groups):
        unique_groups = np.unique(groups)
        transformed = np.zeros_like(raw_pred)
        
        for group_id in unique_groups:
            mask = groups == group_id
            if np.sum(mask) > 0:
                transformed[mask] = softmax(raw_pred[mask])
                
        return transformed
    
    def set_weights(self, weights):
        self._weights = np.array(weights)
        self._model.coef_ = np.array(weights)
        self._model.intercept_ = np.zeros((1,))
        self._fitted = True
        

