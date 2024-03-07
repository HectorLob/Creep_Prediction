import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from collections.abc import Callable



class ModelBuilder:
    def __init__(self, parameters: dict):
        """Class to build a model based on the parameters dict in parameters.yml
        """
        self.model_name = parameters['model']
        self.regressor = self.create_regressor()
        self.hyperparameters = parameters['hyperparameters'][self.model_name]
        self.metric = parameters['metric']
        self.scoring, self.scoring_function, self.higher_is_better = self.select_metric()
        self.n_iter = parameters['n_iter']
    

    def create_regressor(self) -> BaseEstimator:
        """Return the regression model for the current configuration
        """
        model_dict = {
            'RandomForestRegressor': RandomForestRegressor,
            'LGBMRegressor': LGBMRegressor,
            'XGBRegressor': XGBRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'LinearRegression': LinearRegression,
            'Lasso': Lasso,
            'Ridge': Ridge,
            }

        assert self.model_name in list(model_dict.keys()), f"Model {self.model_name} not implemented. Implemented models are {list(model_dict.keys())}"

        return model_dict[self.model_name]()
    

    def select_metric(self) -> tuple[str, Callable, bool]:
        """
        Select the appropriate metrics for model evaluation.

        Args:
        -----
        - metric (str): Metric to use for model evaluation. (parameters['metric'])

        Returns:
        --------
        - scoring (str): Scoring metric string for RandomizedSearchCV.
        - scoring_function (function): Function to compute the selected metric.
        - higher_is_better (bool): Whether the selected metric should be maximized or minimized.
        """
        metric = self.metric
        implemented_metrics = ['r2', 'mse', 'mae', 'rmse', 'mape']
        assert metric in implemented_metrics, f"Metric {metric} not implemented. Implemented metrics are {implemented_metrics}"

        if metric == 'r2':
            scoring = 'r2'
            scoring_function = r2_score
            higher_is_better = True

        elif metric == 'mse':
            scoring = 'neg_mean_squared_error'
            scoring_function = mean_squared_error
            higher_is_better = False

        elif metric == 'mae':
            scoring = 'neg_mean_absolute_error'
            scoring_function = mean_absolute_error
            higher_is_better = False

        elif metric == 'rmse':
            scoring = 'neg_root_mean_squared_error'
            scoring_function = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            higher_is_better = False

        elif metric == 'mape':
            scoring = 'neg_mean_absolute_percentage_error'
            scoring_function = mean_absolute_percentage_error
            higher_is_better = False

        return scoring, scoring_function, higher_is_better
    


def create_model_builder(parameters: dict) -> ModelBuilder:
    """Create a ModelBuilder object based on conf/base/parameters.yml
    """
    return ModelBuilder(parameters)
