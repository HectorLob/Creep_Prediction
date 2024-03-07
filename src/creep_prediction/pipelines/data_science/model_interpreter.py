import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


#############################################################################################################
# Sept. 18, 2023
# Interpretability: track feature importances
#############################################################################################################


class ModelInterpreter():
    def __init__(self, model):
        self.model = model
        self.model_name = self.return_model_name()  # == parameters['model']
        self.model_params = self.return_model_params()
        self.feature_names = self.return_feature_names()
        self.feature_importances = self.return_feature_importances()
        self.df = self.return_df()

    def unpickle_model(self):
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)        
        
    def return_model_name(self):
        return self.model.__class__.__name__
    
    def return_model_params(self):
        return self.model.get_params()
    
    def return_feature_names(self):

        if self.model_name == 'XGBRegressor':
            return self.model.get_booster().feature_names
        elif self.model_name == 'LGBMRegressor':
            return self.model.feature_name_
        else:
            return self.model.feature_names_in_
        
    def return_feature_importances(self):
        if self.model_name == 'XGBRegressor':
            importance_dict = self.model.get_booster().get_score(importance_type='gain')
            importance_values = list(importance_dict.values())
            normalized_importances = [float(i)/sum(importance_values) for i in importance_values]
            return normalized_importances
        
        elif self.model_name == 'LGBMRegressor':
            importance_values = self.model.feature_importances_
            normalized_importances = [float(i)/sum(importance_values) for i in importance_values]
            return normalized_importances
        
        elif self.model_name == 'RandomForestRegressor' or self.model_name == 'DecisionTreeRegressor':
            return list(self.model.feature_importances_)
        
        elif self.model_name in ['LinearRegression', 'Lasso', 'Ridge']:
            # This is not the same as feature importances, but it is a way to interpret the model
            coefs = self.model.coef_
            sum_abs_coefs = np.sum(np.abs(coefs)) # Calculate the sum of absolute values of coefficients
            normalized_coefs = np.abs(coefs) / sum_abs_coefs # Normalize the coefficients to make them sum up to 1
            return normalized_coefs
        
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented. \n"
                                      "Implemented models are XGBRegressor, LGBMRegressor, \n"
                                      "RandomForestRegressor, DecisionTreeRegressor, \n"
                                      "LinearRegression, Lasso, Ridge")

        
    def return_df(self):
        feature_dict = dict(zip(self.feature_names, self.feature_importances))
        # sort dict by keys
        #feature_dict = {k: v for k, v in sorted(feature_dict.items(), key=lambda item: item[0])}
        return pd.DataFrame.from_dict(feature_dict, orient='index')#, columns=[model_name])
