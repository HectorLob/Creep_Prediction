import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, GroupKFold
from sklearn.base import BaseEstimator

from .model_builder import *
from .utils import *


def _train_model(model_builder: ModelBuilder, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_dev: pd.DataFrame, y_dev: pd.Series, random_state: int) -> RandomizedSearchCV:

    """
    Train a model using RandomizedSearchCV with a predefined train/dev split.
    We do this to avoid data leakage, i.e. having rows of the same groupby_col in both splits simultaneously.

    Args:
    ------
    - model_builder (ModelBuilder): An instance of the ModelBuilder class.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target variable.
    - X_dev (pd.DataFrame): Development features.
    - y_dev (pd.Series): Development target variable.
    - random_state (int): Seed for the random number generator, for reproducibility.

    Returns:
    --------
    - random_search (RandomizedSearchCV): Trained model as an instance of RandomizedSearchCV.
    """
    X = pd.concat([X_train, X_dev]) 
    y = pd.concat([y_train, y_dev])

    fold = np.concatenate([-np.ones(len(X_train)), np.zeros(len(X_dev))]) # array full of -1 to indicate train set and 0 (dev set)

    random_search = RandomizedSearchCV(estimator=model_builder.regressor,
                                       param_distributions=model_builder.hyperparameter_grid,
                                       n_iter=model_builder.n_iter,
                                       scoring=model_builder.scoring, 
                                       n_jobs=-1,
                                       cv=PredefinedSplit(fold), 
                                       random_state=random_state,
                                       refit=True) # It is a default value, just to be explicit
    random_search.fit(X, y)  

    return random_search



def _perform_inner_cv_iteration(df_traindev: pd.DataFrame, train_idx: list, dev_idx: list,
                                info_df: pd.DataFrame, features: list, target: str,
                                model_builder: ModelBuilder, parameters: dict,
                                best_score: float, best_model: BaseEstimator,
                                k_fold_outer: int, k_fold_inner: int) -> tuple:
    """
    This function performs the inner cv loop within nested_cv function.
    It trains the model with the specified parameters and evaluates it. 
    If the model's performance is better than the best score so far, 
    the function updates the best score and the best model.
    If parameters['track_inner'] is True, it tracks dev_score to wandb.
    
    Args:
    ------
    - df_traindev (pd.DataFrame): The training and development data combined into one DataFrame.
    - train_idx (list or array-like): Indices for training data.
    - dev_idx (list or array-like): Indices for development (validation) data.
    - info_df (pd.DataFrame): Additional information data. 
    - features (list of str)
    - target (str)
    - model_builder (ModelBuilder): An instance of the ModelBuilder class. 
    - parameters (dict): conf/base/parameters.yml parameter file
    - best_score (float): The best score achieved so far in the cv process.
    - best_model (object): The best model trained so far.
    - k_fold_outer (int): The current fold number of the outer cv loop.
    - k_fold_inner (int): The current fold number of the inner cv loop.

    Returns:
    --------
    - best_model (object): The best model trained so far.
    - best_score (float): The best score achieved so far.
    """
    df_train = df_traindev.iloc[train_idx]
    df_dev = df_traindev.iloc[dev_idx]
                        
    X_train, y_train = df_train[features], df_train[target]
    X_dev, y_dev = df_dev[features], df_dev[target]

    random_search = _train_model(model_builder, X_train, y_train, X_dev, y_dev,
                                 random_state=parameters['random_state'])
    
    y_dev_pred = random_search.predict(X_dev)

    if target.startswith('R_'): # If target is a ratio, convert predicted ratio to CM
        y_dev, y_dev_pred = predicted_ratio_to_CM(y_dev, y_dev_pred, target, info_df)

    dev_score = model_builder.scoring_function(y_dev, y_dev_pred)

    higher_is_better = model_builder.higher_is_better

    # Check if `dev_score` is better than `best_score` according to the chosen metric ('r2' requires higher values; the rest, lower)
    if (higher_is_better and dev_score > best_score) or (not higher_is_better and dev_score < best_score):
        best_score = dev_score
        best_model = random_search.best_estimator_

    if parameters['track_inner'] and parameters['use_wandb']:

        data_info_dict = {'train_size': df_train.shape[0],
                          'dev_size': df_dev.shape[0],
                          'dev_frac': df_dev.shape[0] / (df_train.shape[0] + df_dev.shape[0]),
                          'train_groups': df_train[parameters['groupby_col']].unique(),
                          'dev_groups': df_dev[parameters['groupby_col']].unique()
                          }
        
        wandb_inner_fold(parameters, data_info_dict,
                         dev_score, k_fold_outer, k_fold_inner,
                         best_hyperparameters=random_search.best_params_)
        
    else:
        print(f'- Dev {model_builder.metric}: {dev_score:.5f} \n')
    
    return best_model, best_score



def nested_cv(data: pd.DataFrame, info: pd.DataFrame,
              parameters: dict) -> tuple[BaseEstimator, dict]:
    """
    Perform a nested cross-validation (cv) on the given data.

    Args:
    -----
    - data (pd.DataFrame): The input data to perform cv on. Last column must be the target variable.
    - info (pd.DataFrame): Additional data with the same index as data.
    - parameters (dict): conf/base/parameters.yml parameter file

    Returns:
    --------
    A tuple containing two elements:
    - The best model found during the cv process.
    - A dictionary containing various metrics and information about the cv process.
    """

    # Shuffle data, and info accordingly
    df = data.copy().sample(frac=1, random_state=parameters['random_state'])
    info_df = info.copy().loc[df.index, :]

    target = df.columns[-1] # the last column is the target variable
    features = list(df.columns[:-1]) # the rest but groupby_col are features
    groupby_col = parameters['groupby_col'] # the column to group by for splitting the data
    features.remove(groupby_col) 

    print(f'Prediciting {target} using {features}')

    model_builder = create_model_builder(parameters) # create a ModelBuilder object

    scoring_function = model_builder.scoring_function 
    higher_is_better = model_builder.higher_is_better

    # Initialize best final score and model
    best_final_score = -np.inf if higher_is_better else np.inf 
    best_final_model = None

    outer_cv = GroupKFold(n_splits=parameters['k_fold_outer'])
    inner_cv = GroupKFold(n_splits=parameters['k_fold_inner'])

    score_error_dict_list = [] # list of dictionaries containing scores and errors for each outer fold

    # Outer loop of the nested cv
    for i, (traindev_idx, test_idx) in enumerate(outer_cv.split(df, groups=df[groupby_col])):

        df_traindev, df_test = df.iloc[traindev_idx], df.iloc[test_idx]
        
        print(f'\n ***Outer loop iter. {i+1}/{parameters["k_fold_outer"]}*** \n')

        # Initialize best score and model
        best_score = -np.inf if higher_is_better else np.inf
        best_model = None

        # Inner loop of the nested cv
        for j, (train_idx, dev_idx) in enumerate(inner_cv.split(df_traindev, groups=df_traindev[groupby_col])):

            print(f'Inner loop iter. {j+1}/{parameters["k_fold_inner"]} \n')
            
            best_model, best_score = _perform_inner_cv_iteration(df_traindev, train_idx, dev_idx,
                                                                 info_df, features, target, model_builder,
                                                                 parameters, best_score, best_model,
                                                                 k_fold_outer=i, k_fold_inner=j)

        X_test, y_test = df_test[features], df_test[target]

        y_test_pred = best_model.predict(X_test)

        if target.startswith('R_'): # If target is a ratio, convert predicted ratio to CM
            y_test_pred_R = y_test_pred
            y_test, y_test_pred = predicted_ratio_to_CM(y_test, y_test_pred, target, info_df)
            analysis_df = return_analysis_df(y_test, y_test_pred, info_df, target, k_fold_test=i, y_pred_R=y_test_pred_R)
        
        else:
            analysis_df = return_analysis_df(y_test, y_test_pred, info_df, target, k_fold_test=i)

        test_score = scoring_function(y_test, y_test_pred)

        # Check if `test_score` is better than `best_final_score` according to the chosen metric ('r2' requires higher values; the rest, lower)
        if (higher_is_better and test_score > best_final_score) or (not higher_is_better and test_score < best_final_score):
            best_final_score = test_score
            best_final_model = best_model

        best_model_hyps = {h: best_model.get_params()[h] for h in model_builder.hyperparameter_grid} # only save tuned hyperparameters

        score_error_dict = return_test_tracking_dict(y_test, y_test_pred, analysis_df) 
        score_error_dict_list.append(score_error_dict)

        if parameters['use_wandb']:
            wandb_outer_fold(parameters, df_traindev, df_test, best_model_hyps, score_error_dict, analysis_df, k_fold_outer=i)
        
        else:
            print(f'Test {model_builder.metric}: {test_score:.4f} \n \n')
  
    final_score_dict = return_final_track_dict(score_error_dict_list, parameters) # model evaluation metrics
    
    if parameters['use_wandb']:
        wandb_final_results(final_score_dict, parameters, best_final_model, model_builder)

    else:
        print(final_score_dict)
        plot_predictions(final_score_dict['final_analysis_df'], parameters)

    return best_final_model, final_score_dict