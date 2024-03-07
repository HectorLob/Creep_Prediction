import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, GroupKFold
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from .model_builder import *
from .model_interpreter import *
from .utils import *



def _perform_inner_cv_iteration(df_traindev: pd.DataFrame, 
                                train_idx: list, 
                                dev_idx: list,
                                info_df: pd.DataFrame, 
                                features: list, 
                                target: str,
                                model_builder: ModelBuilder, 
                                parameters: dict,
                                best_score: float, 
                                best_model: BaseEstimator) -> tuple[BaseEstimator, float]:
    """
    This function performs the inner cv loop within nested_cv function.
    First, it scales the data and then it trains a model using RandomizedSearchCV with a predefined train/dev split.
    If the score of the model on the dev set is better than the best score achieved so far, it updates the best score and model.
    
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

    Returns:
    --------
    - best_model (object): The best model trained so far.
    - best_score (float): The best score achieved so far.
    """
    df_train_inner = df_traindev.iloc[train_idx]
    df_dev = df_traindev.iloc[dev_idx]
                        
    X_train_inner = df_train_inner[features]
    y_train_inner = df_train_inner[target]
        
    X_dev = df_dev[features]
    y_dev = df_dev[target]

    X = pd.concat([X_train_inner, X_dev])
    y = pd.concat([y_train_inner, y_dev])

    fold = np.concatenate([-np.ones(len(X_train_inner)), np.zeros(len(X_dev))]) # array full of -1 to indicate train set and 0 (dev set)

    random_search = RandomizedSearchCV(estimator=model_builder.regressor,
                                       param_distributions=model_builder.hyperparameters,
                                       n_iter=model_builder.n_iter,
                                       scoring=model_builder.scoring, 
                                       n_jobs=-1, # use all available processors
                                       cv=PredefinedSplit(fold), 
                                       random_state=parameters['random_state'],
                                       refit=True) # True by default
    random_search.fit(X, y) 
    
    # Make predictions on the scaled dev data
    y_dev_pred = random_search.predict(X_dev)

    if target.startswith('R_'): # If target is a ratio, convert predicted ratio to CM
        y_dev, y_dev_pred = predicted_ratio_to_CM(y_dev, y_dev_pred, target, info_df)

    dev_score = model_builder.scoring_function(y_dev, y_dev_pred)

    higher_is_better = model_builder.higher_is_better

    # Check if `dev_score` is better than `best_score` according to the chosen metric ('r2' requires higher values; the rest, lower)
    if (higher_is_better and dev_score > best_score) or (not higher_is_better and dev_score < best_score):
        best_score = dev_score
        best_model = random_search.best_estimator_

    return best_model, best_score



def main(data: pd.DataFrame, 
         info: pd.DataFrame,
         parameters: dict) -> tuple[BaseEstimator, dict]:
    """
    Main function for the data science pipeline.
    If hyperparameters are to be tuned (parameters['tune_hyps'] == True), it performs a nested cv,
    Otherwise, it performs a k-fold cv with the hyperparameters specified in the parameters.yml file.

    Args:
    ------
    - data (pd.DataFrame): The data to be used for training and testing.
    - info (pd.DataFrame): Additional information data.
    - parameters (dict): conf/base/parameters.yml parameter file

    Returns:
    --------
    - best_final_model (object): The best performing model
    - final_score_dict (dict): Dictionary containing the final scores and errors.
    - feature_importances_final_df (pd.DataFrame): Dataframe containing features importances for each outer fold.

    """
    # Shuffle data, and info accordingly
    df = data.copy().sample(frac=1, random_state=parameters['random_state'])
    info_df = info.copy().loc[df.index, :]

    validate_parameters(parameters) # validate parameters in parameters.yml file

    target = parameters['target']
    features = parameters['features']
    groupby_col = parameters['groupby_col'] # the column to group by for splitting the data

    model_builder = create_model_builder(parameters) # create a ModelBuilder object

    scoring_function = model_builder.scoring_function 
    regressor = model_builder.regressor
    higher_is_better = model_builder.higher_is_better

    print(f'Prediciting {target} with {features} using {model_builder.model_name} \n')

    # Initialize best final score and model
    best_final_score = -np.inf if higher_is_better else np.inf 
    best_final_model = None

    if parameters['tune_hyps']: # if tune_hyps is True, we perform a nested cv
        cv = GroupKFold(n_splits=parameters['k_fold_outer']) # outer loop of the nested cv
        inner_cv = GroupKFold(n_splits=parameters['k_fold_inner'])

    else: # if tune_hyps is False, we perform a k-fold cv
        cv = GroupKFold(n_splits=parameters['k_fold'])

    score_error_dict_list = [] # list of dictionaries containing scores and errors for each outer fold
    feature_importances_final_df = pd.DataFrame() # dataframe containing features importances for each outer fold
    params_final_dict = {} # dictionary containing the final parameters for each fold
    lr_coef_list = [] # list of lists containing the coefficients of the linear regression model for each outer fold

    for i, (train_idx, test_idx) in enumerate(cv.split(df, groups=df[groupby_col]), start=1):

        print(f'\n ***k-fold {i}/{parameters["k_fold_outer"]}*** \n')

        df_train = df.iloc[train_idx] # if cv_type == 'nested', this is the train+dev set
        df_test = df.iloc[test_idx]

        X_train = df_train[features]
        y_train = df_train[target]

        X_test = df_test[features]
        y_test = df_test[target]
        
        if parameters['tune_hyps']:

            # Initialize best score and model
            best_score = -np.inf if higher_is_better else np.inf
            best_model = None

            # Inner loop of the nested cv
            for j, (train_idx, dev_idx) in enumerate(inner_cv.split(df_train, groups=df_train[groupby_col]), start=1):

                print(f'Inner loop iter. {j}/{parameters["k_fold_inner"]} \n')
                
                best_model, best_score = _perform_inner_cv_iteration(df_train, train_idx, dev_idx,
                                                                     info_df, features, target, model_builder,
                                                                     parameters, best_score, best_model)
                
            regressor = best_model # This model is optimized on the dev set and fitted on the train+dev set
            
        else: # if tune_hyps is False, we perform a k-fold cv with the hyps. specified in the parameters.yml file
            if model_builder.hyperparameters: # if hyps. are not empty, we use the specified hyps. Else, default ones
                regressor.set_params(**model_builder.hyperparameters)
            print(regressor.get_params())

            regressor.fit(X_train, y_train) # Fit the model on the training data

        # Make predictions 
        y_pred = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)

        if target.startswith('R_'): # If target is a ratio, convert predicted ratio to CM
            CM_true_test, CM_pred_test = predicted_ratio_to_CM(y_test, y_pred, target, info_df)
            analysis_df_test = return_analysis_df(CM_true_test, CM_pred_test, info_df, target, k_fold_test=i, y_pred_R=y_pred)
            CM_true_train, CM_pred_train = predicted_ratio_to_CM(y_train, y_pred_train, target, info_df)
            analysis_df_train = return_analysis_df(CM_true_train, CM_pred_train, info_df, target, k_fold_test=i, y_pred_R=y_pred_train)
        
        else:
            CM_true_test, CM_pred_test = y_test, y_pred
            analysis_df_test = return_analysis_df(CM_true_test, CM_pred_test, info_df, target, k_fold_test=i)
            CM_true_train, CM_pred_train = y_train, y_pred_train
            analysis_df_train = return_analysis_df(CM_true_train, CM_pred_train, info_df, target, k_fold_test=i)

        test_score = scoring_function(CM_true_test, CM_pred_test)
        print(f'Test {model_builder.metric}: {test_score:.4f} \n \n')
        print(f'Train {model_builder.metric}: {scoring_function(CM_true_train, CM_pred_train):.4f} \n \n')

        if (higher_is_better and test_score > best_final_score) or (not higher_is_better and test_score < best_final_score):
            best_final_score = scoring_function(CM_true_test, CM_pred_test)
            best_final_model = regressor

        # Save scores and errors for each outer fold
        score_error_dict = return_test_tracking_dict(CM_true_test, CM_pred_test, 
                                                     CM_true_train, CM_pred_train, 
                                                     analysis_df_test, analysis_df_train)
         
        score_error_dict_list.append(score_error_dict)

        # Save feature importances for each outer fold
        feature_importances_df = ModelInterpreter(regressor).df
        feature_importances_df.columns = [f'fold_{i}']
        feature_importances_final_df = pd.concat([feature_importances_final_df, feature_importances_df], axis=1)
        
        # Save final parameters for each outer fold
        if model_builder.model_name in ['LinearRegression', 'Ridge', 'Lasso']:
            params_dict = {'coef': regressor.coef_,
                           'intercept': regressor.intercept_}
            lr_coef_list.append(params_dict)
            if model_builder.model_name != 'LinearRegression':
                params_dict['alpha'] = regressor.alpha # save alpha for Ridge and Lasso
        else:
            # Track only tuned or defined hyperparameters
            params_dict = {h: regressor.get_params()[h] for h in model_builder.hyperparameters} if model_builder.hyperparameters else {}

        params_final_dict[f'fold_{i}'] = params_dict
  
    hyp_df = pd.DataFrame(params_final_dict).T # df containing the selected hyperparameters for each outer fold

    final_score_dict = return_final_track_dict(score_error_dict_list) # model evaluation metrics
    final_score_dict['model_name'] = model_builder.model_name
    final_score_dict['conf'] = parameters['tags']

    if parameters['use_wandb']:
        wandb_final_results(parameters, final_score_dict, feature_importances_final_df, hyp_df, feature_importances_final_df)

    else:
        plot_predictions(final_score_dict['final_analysis_df_test'], parameters)
        plot_feature_importances(feature_importances_final_df, parameters)
        score_dict = {k: v for k, v in final_score_dict.items() if k not in ['final_analysis_df_train', 'final_analysis_df_test']}
        print(pd.Series(score_dict))

    #print(lr_coef_list)

    if model_builder.model_name == 'LinearRegression':
        mean_coefs = np.mean([lr['coef'] for lr in lr_coef_list], axis=0)
        mean_intercept = np.mean([lr['intercept'] for lr in lr_coef_list], axis=0)
        std_coefs = np.std([lr['coef'] for lr in lr_coef_list], axis=0)
        std_intercept = np.std([lr['intercept'] for lr in lr_coef_list], axis=0)

        # Print coefs for each feature and intercept
        print(f'Intercept: {np.round(mean_intercept, 4)} +/- {np.round(std_intercept, 4)}')
        for f, c, s in zip(features, mean_coefs, std_coefs):
            print(f'{f}: {np.round(c, 4)} +/- {np.round(s, 4)}')

        # Evaluate mean coefficients and intercept with ALL DATA:
        X = df[features]
        y = df[target]
        y_pred = mean_coefs @ X.T + mean_intercept

        if target.startswith('R_'):
            y, y_pred = predicted_ratio_to_CM(y, y_pred, target, info_df)

        analysis_df_last = return_analysis_df(y, y_pred, info_df, target, k_fold_test='all', y_pred_R=y_pred)
        #analysis_df_last = analysis_df_last.sort_index() #sort by index number

        print(r2_score(y, y_pred))
        print(analysis_df_last)

    return best_final_model, final_score_dict
