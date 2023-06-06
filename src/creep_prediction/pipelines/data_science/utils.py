import wandb
import pandas as pd
import numpy as np
import uuid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



def map_ratio_to_CM(target: str) -> tuple[str, str]:
    """
    Args:
    -----
    - target (str): The target column name

    Returns:
    --------
    - CM_predict, CM_before (tuple[str, str]): column names of the predicted and the previous CMs, respectively.
    """
    ratio_CM_dict = {
        'R_10/1': {'CM_predict': 'CM_10', 'CM_before': 'CM_1'},
        'R_100/10': {'CM_predict': 'CM_100', 'CM_before': 'CM_10'},
        'R_1000/100': {'CM_predict': 'CM_1000', 'CM_before': 'CM_100'},
        'R_10000/1000': {'CM_predict': 'CM_10000', 'CM_before': 'CM_1000'},  
        'R_100000/10000': {'CM_predict': 'CM_100000', 'CM_before': 'CM_10000'}
    }
    CM_predict = ratio_CM_dict[target]['CM_predict']
    CM_before = ratio_CM_dict[target]['CM_before']

    return CM_predict, CM_before



def predicted_ratio_to_CM(y_true_ratio: pd.Series, y_pred_ratio: pd.Series, 
                          target: str, CM_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Obtain the true and predicted CM values from the true and predicted ratio values and the info df.

    Args:
    -----
    - y_true_ratio (pd.Series): True ratio values (y_dev or y_test)
    - y_pred_ratio (pd.Series): Predicted ratio values.
    - target (str): Target variable name, either starting with 'CM' or 'R_'.
    - CM_df (pd.DataFrame): DataFrame containing the true CM values.

    Returns:
    --------
    - CM_true, CM_pred (tuple[pd.Series, pd.Series]): True and predicted CM values.
    """
    # Get the corresponding CM column names
    CM_predict_str, CM_before_str = map_ratio_to_CM(target)

    # Get true CM values from info df
    CM_true = CM_df.loc[y_true_ratio.index, CM_predict_str]
    CM_before = CM_df.loc[y_true_ratio.index, CM_before_str]

    # Get predicted CM values
    CM_pred = y_pred_ratio * CM_before # e.g. CM_1000 = R_1000/100 * CM_100

    return CM_true, CM_pred



def _return_experiment_name(parameters: dict) -> str:
    """Return the name of the experiment. If the experiment_name parameter is not specified in the parameters.yml file,
    a random string of 5 characters is returned.
    """
    if parameters['experiment_name']:
        return parameters['experiment_name']
    else:
        return str(uuid.uuid4())[:5] # Return a random string of 5 characters


def _return_config(parameters: dict) -> dict:
    """Return the parameters of the current experiment configuration.
    """
    keys = ['target', 'model', 'features', 'feature_engineering', 'metric', 'random_state', 'n_iter', 'groupby_col']
    return {key: parameters[key] for key in keys}



def wandb_inner_fold(parameters: dict, data_info_dict: dict, dev_score: float, 
                     k_fold_outer: int, k_fold_inner: int, best_hyperparameters: dict):
    """
    Track the results of the inner fold of the cross-validation.
    
    Args:
    -----
    - parameters (dict): conf/base/parameters.yml parameter file
    - data_info_dict (dict): dictionary containing information about the data
    - k_fold_outer (int): The current fold number of the outer cv loop.
    - k_fold_inner (int): idem for the inner cv loop.
    - dev_score (float): score of the model on the development set
    - best_hyperparameters (RandomizedSearchCV.best_params_): best hyperparameters found by the RandomizedSearchCV

    Returns:
    --------
    None. The function logs the results directly to the W&B dashboard.

    """
    experiment_name = _return_experiment_name(parameters)

    run = wandb.init(project=parameters['project_name'], 
                     group=f'Outer {k_fold_outer}',
                     name=f"{experiment_name}: Outer {k_fold_outer + 1} - Inner {k_fold_inner + 1}",
                     notes=parameters['notes'],
                     config={**_return_config(parameters), **best_hyperparameters, **data_info_dict},
                     )
    
    wandb.log({'dev_score': dev_score})

    run.finish()
    


def wandb_outer_fold(parameters: dict, df_traindev: pd.DataFrame, df_test: pd.DataFrame, 
                     best_model_hyps: dict, score_error_dict: dict, analysis_df: pd.DataFrame, k_fold_outer: int):
    """
    Track the results of the outer fold of the cross-validation.

    Args:
    -----
    - parameters (dict): conf/base/parameters.yml parameter file.
    - df_traindev (pd.DataFrame): training and development data.
    - df_test (pd.DataFrame): test set.
    - best_model_hyps (dict): best hyperparameters found by the model.
    - score_error_dict (dict): Dictionary consisting of scoring and error metrics.
    - analysis_df (pd.DataFrame): DataFrame containing analysis data.
    - k_fold_outer (int): The current fold number of the outer cv loop.

    Returns:
    --------
    - None. The function logs the results directly to the W&B dashboard.
    """
    data_info_dict = {'traindev_size': df_traindev.shape[0],
                    'test_size': df_test.shape[0],
                    'test_frac': df_test.shape[0] / (df_traindev.shape[0] + df_test.shape[0]),
                    'traindev_groups': df_traindev[parameters['groupby_col']].unique(),
                    'test_groups': df_test[parameters['groupby_col']].unique(),
                    }
    
    experiment_name = _return_experiment_name(parameters)

    run = wandb.init(project=parameters['project_name'], 
                     group='Outer',
                     name=f"{experiment_name}: Outer {k_fold_outer + 1} - Test",
                     notes=parameters['notes'],
                     config={**_return_config(parameters), **best_model_hyps, **data_info_dict},
                     )

    wandb.log(score_error_dict)
    
    # Log analysis dataframe
    wandb.log({'analysis_df': wandb.Table(dataframe=analysis_df.reset_index(), allow_mixed_types=True)})
    
    run.finish()  



def wandb_final_results(final_score_dict: dict, parameters: dict, 
                        best_final_model, model_builder):
    """
    Track and log the final results of the model.

    Args:
    -----
    - final_score_dict (dict): Dictionary of scoring metrics.
    - parameters (dict): conf/base/parameters.yml parameter file.
    - best_final_model (BaseEstimator): The best model found during the experiment.
    - model_builder (ModelBuilder): An instance of the ModelBuilder class.

    Returns:
    --------
    - None. The function logs the results directly to the W&B dashboard.
    """
    experiment_name = _return_experiment_name(parameters)

    final_model_hyps = {h: best_final_model.get_params()[h] for h in model_builder.hyperparameter_grid}

    run = wandb.init(project=parameters['project_name'],
                     name=f"{experiment_name}: Final results",
                     group='Final results',
                     notes=parameters['notes'],
                     config={**_return_config(parameters), **final_model_hyps} 
                     )
    
    # Log analysis dataframe
    final_analysis_df = final_score_dict['final_analysis_df']
    wandb.log({'final_analysis_df': wandb.Table(dataframe=final_analysis_df.reset_index().sort_values(by=final_analysis_df.columns[-1]), 
                                                allow_mixed_types=True)})

    # List of keys to remove from the dictionary before logging
    keys_to_remove = ['final_analysis_df', 'config', 'experiment_name']

    wandb.log({k: v for k, v in final_score_dict.items() if k not in keys_to_remove})

    run.finish()



def return_analysis_df(y_true: pd.Series, y_pred: pd.Series,
                       info_df: pd.DataFrame, target: str,
                       k_fold_test: int,
                       y_pred_R: pd.Series = None) -> pd.DataFrame:
    """
    Return a DataFrame with info columns, CM columns, predicted CM and relative error %.

    Args:
    -----
    - y_true (pd.Series): True values.
    - y_pred (pd.Series): Predicted values.
    - info_df (pd.DataFrame): DataFrame containing the info columns and the CM columns.
    - target (str): The target variable name, either starting with 'CM' or 'R_'.
    - k_fold_test (int): The number of the k-fold test set.
    - y_pred_R (pd.Series): Predicted R values. Only needed if target starts with 'R_'.

    Returns:
    --------
    - analysis_df (pd.DataFrame): DataFrame with info columns, CM columns, predicted CM and relative error %.
    """
    analysis_df = info_df.copy().loc[y_true.index, :] # only keep the rows within the test set

    if target.startswith('R_'):
        target_CM, _ = map_ratio_to_CM(target) # get the corresponding CM column name
        R_cols = [col for col in analysis_df.columns if col.startswith('R_')]
        drop_R_cols = [col for col in R_cols if col != target] # drop all R columns except the target
        analysis_df.drop(drop_R_cols, axis=1, inplace=True)
        # Add predicted R_ values
        analysis_df[target + '_pred'] = y_pred_R
    else:
        target_CM = target

    CM_cols = [col for col in analysis_df.columns if col.startswith('CM_')]  
    target_CM_idx = CM_cols.index(target_CM) # index of the target CM column
    # drop all CM columns greater than the target
    drop_CM_cols = [col for col in CM_cols if CM_cols.index(col) > target_CM_idx]
    analysis_df = analysis_df.drop(drop_CM_cols, axis=1)

    # Add k_fold_test column
    analysis_df['k_fold_test'] = k_fold_test

    # Add predicted CM values
    pred_col = target_CM + '_pred'
    analysis_df[target_CM + '_pred'] = y_pred
    
    # Compute relative error, positive or negative, and absolute value
    analysis_df['+-Rel_error_%'] = (analysis_df[target_CM] - analysis_df[pred_col]) / analysis_df[target_CM] * 100
    analysis_df['Rel_error_%'] = analysis_df['+-Rel_error_%'].abs() 

    # Reorder columns
    R_cols = [col for col in analysis_df.columns if col.startswith('R_')] if target.startswith('R_') else []
    CM_cols = [col for col in analysis_df.columns if col.startswith('CM_')]
    error_cols = ['+-Rel_error_%', 'Rel_error_%']
    other_cols = [col for col in analysis_df.columns if col not in R_cols + CM_cols + error_cols]
    analysis_df = analysis_df[other_cols + R_cols + CM_cols + error_cols]

    # Sort by relative error
    return analysis_df.sort_values(by=analysis_df.columns[-1], ascending=False)



def return_test_tracking_dict(y_test: pd.Series, y_test_pred: pd.Series, analysis_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate and return a dictionary of scoring metrics for a single test set.

    Args:
    -----
    - y_test (pd.Series): The true target values for the test set.
    - y_test_pred (pd.Series): The predicted target values for the test set.
    - analysis_df (pd.DataFrame): A DataFrame containing additional analysis of the test set predictions.

    Returns:
    --------
    A dictionary containing the following scoring metrics:
    - r2_test (float): The R-squared score for the test set.
    - rmse_test (float): The root mean squared error.
    - mae_test (float): The mean absolute error.
    - mape_test (float): The mean absolute percentage error.
    - max_error_test (float): The maximum relative error.
    - median_error_test (float): The median relative error.
    - mean_error_test (float): The mean relative error.
    - mean_error_abs_test (float): The mean absolute relative error.
    - perc_error_lt_10_test (float): The percentage of relative errors less than 10%.
    - perc_error_lt_5_test (float): The percentage of relative errors less than 5%.
    """
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    test_max_error = analysis_df['Rel_error_%'].max()
    test_median_error = analysis_df['Rel_error_%'].median()
    test_mean_error = analysis_df['+-Rel_error_%'].mean()
    test_mean_error_abs = analysis_df['Rel_error_%'].mean()

    perc_error_lt_10 = len(analysis_df[analysis_df['Rel_error_%'] < 10]) / len(analysis_df) * 100 # lt = less than
    perc_error_lt_5 = len(analysis_df[analysis_df['Rel_error_%'] < 5]) / len(analysis_df) * 100

    return {'r2_test': r2_test,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'mape_test': mape_test,
            'max_error_test': test_max_error,
            'median_error_test': test_median_error,
            'mean_error_test': test_mean_error,
            'mean_error_abs_test': test_mean_error_abs,
            'perc_error_lt_10_test': perc_error_lt_10,
            'perc_error_lt_5_test': perc_error_lt_5,
            'analysis_df': analysis_df}



def return_final_track_dict(test_track_dict_list: list[dict[str, float]],
                            parameters: dict) -> dict[str, float]:
    """
    Calculate and return a dictionary of final scoring metrics for a list of test sets.

    Args:
    -----
    - test_track_dict_list (list[dict[str, float]]): A list of dictionaries containing scoring metrics for each test set.
    - parameters (dict): conf/base/parameters.yml parameter file

    Returns:
    --------
    A dictionary containing the following final scoring metrics:
    - R2 mean (float): The mean R-squared score across all test sets.
    - R2 std (float): The standard deviation of the R-squared scores across...
    - RMSE mean (float): The mean root mean squared error...
    - RMSE std (float): The standard deviation of the root mean squared errors...
    - MAE mean (float): The mean mean absolute error...
    - MAE std (float): The standard deviation of the mean absolute errors...
    - MAPE mean (float): The mean mean absolute percentage error across all test sets.
    - MAPE std (float): The standard deviation of the mean absolute percentage errors...
    - Max error % (float): The maximum relative error...
    - Mean median error % (float): The mean median relative error...
    - Mean error % (float): The mean relative error...
    - Mean abs error % (float): The mean absolute relative error...
    - Mean % < 5% error (float): The mean percentage of rows with relative errors less than 5% ...
    - Mean % < 10% error (float): The mean percentage of rows with relative errors less than 10% ...
    - final_analysis_df (pd.DataFrame): A DataFrame containing concatenated analysis dfs of the test set predictions.
    """
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    max_errors = []
    median_errors = []
    mean_errors = []
    mean_errors_abs = []
    lt_5 = []
    lt_10 = []

    final_analysis_df = pd.DataFrame()

    for test_track_dict in test_track_dict_list:
        r2_scores.append(test_track_dict['r2_test'])
        rmse_scores.append(test_track_dict['rmse_test'])
        mae_scores.append(test_track_dict['mae_test'])
        mape_scores.append(test_track_dict['mape_test'])
        max_errors.append(test_track_dict['max_error_test'])
        median_errors.append(test_track_dict['median_error_test'])
        mean_errors.append(test_track_dict['mean_error_test'])
        mean_errors_abs.append(test_track_dict['mean_error_abs_test'])
        lt_5.append(test_track_dict['perc_error_lt_5_test'])
        lt_10.append(test_track_dict['perc_error_lt_10_test'])
        final_analysis_df = pd.concat([final_analysis_df, test_track_dict['analysis_df']])

    return {'R2 mean': np.round(np.mean(r2_scores), 5),
            'R2 std': np.round(np.std(r2_scores), 5),
            'RMSE mean': np.round(np.mean(rmse_scores), 5),
            'RMSE std': np.round(np.std(rmse_scores), 5),
            'MAE mean': np.round(np.mean(mae_scores), 5),
            'MAE std': np.round(np.std(mae_scores), 5),
            'MAPE mean': np.round(np.mean(mape_scores), 5),
            'MAPE std': np.round(np.std(mape_scores), 5),
            'Max error %': np.round(max(max_errors), 5),
            'Mean median error %': np.round(np.mean(median_errors), 5),
            'Mean error %': np.round(np.mean(mean_errors), 5),
            'Mean abs error %': np.round(np.mean(mean_errors_abs), 5),
            'Mean % < 5% error': np.round(np.mean(lt_5), 5),
            'Mean % < 10% error': np.round(np.mean(lt_10), 5),
            'final_analysis_df': final_analysis_df,
            'config': _return_config(parameters),
            'experiment_name': _return_experiment_name(parameters)}