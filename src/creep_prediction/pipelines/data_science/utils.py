# Auxiliary functions for the data_science pipeline.

import wandb
import pandas as pd
import numpy as np
import uuid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# use seaborn colorblnd palette in matplotlib
sns.set_palette('colorblind')


def _generate_ratio_CM_dict() -> dict:
    """Generate a dictionary that maps all possible ratio column names to their corresponding CM column names."""
    ratio_CM_dict = {}

    for i in range(6):  # 6 because we have CM_e0 to CM_e5
        for j in range(i):
            ratio = f"R_e{i}/e{j}"
            ratio_CM_dict[ratio] = {'CM_predict': f'CM_e{i}', 'CM_before': f'CM_e{j}'}

    return ratio_CM_dict



def map_ratio_to_CM(target: str) -> tuple[str, str]:
    """
    Args:
    -----
    - target (str): The target column name

    Returns:
    --------
    - CM_predict, CM_before (tuple[str, str]): column names of the predicted and the previous CMs, respectively.
    """
    ratio_CM_dict = _generate_ratio_CM_dict()

    return ratio_CM_dict[target]['CM_predict'], ratio_CM_dict[target]['CM_before']



def predicted_ratio_to_CM(y_true_ratio: pd.Series, 
                          y_pred_ratio: pd.Series, 
                          target: str, 
                          info_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Obtain the true and predicted CM values from the true and predicted ratio values and the info df.

    Args:
    -----
    - y_true_ratio (pd.Series): True ratio values (y_dev or y_test)
    - y_pred_ratio (pd.Series): Predicted ratio values.
    - target (str): Target variable name, either starting with 'CM' or 'R_'.
    - info_df (pd.DataFrame): DataFrame containing the true CM values.

    Returns:
    --------
    - CM_true, CM_pred (tuple[pd.Series, pd.Series]): True and predicted CM values.
    """
    # Get the corresponding CM column names
    CM_predict_str, CM_before_str = map_ratio_to_CM(target)

    # Get true CM values from info df
    CM_true = info_df.loc[y_true_ratio.index, CM_predict_str]
    CM_before = info_df.loc[y_true_ratio.index, CM_before_str]

    # Get predicted CM values
    CM_pred = y_pred_ratio * CM_before # e.g. CM_1000 = R_1000/100 * CM_100

    return CM_true, CM_pred



def return_analysis_df(y_true: pd.Series, 
                       y_pred: pd.Series,
                       info_df: pd.DataFrame, 
                       target: str,
                       k_fold_test: int,
                       y_pred_R: pd.Series=None) -> pd.DataFrame:
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
        analysis_df[target + '_pred'] = y_pred_R # Add predicted R_ values
    else:
        target_CM = target

    CM_cols = [col for col in analysis_df.columns if col.startswith('CM_')]  
    target_CM_idx = CM_cols.index(target_CM) # index of the target CM column
    drop_CM_cols = [col for col in CM_cols if CM_cols.index(col) > target_CM_idx] # drop all CM columns greater than the target
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



def return_test_tracking_dict(y_test: pd.Series, y_test_pred: pd.Series, 
                              y_train: pd.Series, y_train_pred: pd.Series,
                              analysis_df_test: pd.DataFrame,
                              analysis_df_train: pd.DataFrame) -> dict[str, float]:
    """
    Calculate and return a dictionary of scoring metrics for a single test set.

    Args:
    -----
    - y_test (pd.Series): The true target values for the test set.
    - y_test_pred (pd.Series): The predicted target values for the test set.
    - analysis_df (pd.DataFrame): A DataFrame containing additional analysis of the test set predictions.

    Returns:
    --------
    A dictionary containing the following scoring metrics for both the test and the train set:
    - r2(float): The R-squared score for the test set.
    - rmse (float): The root mean squared error.
    - mae (float): The mean absolute error.
    - mape (float): The mean absolute percentage error.
    - max_error (float): The maximum relative error.
    - median_error (float): The median relative error.
    - mean_error (float): The mean relative error.
    - mean_error_abs (float): The mean absolute relative error.
    - perc_error_lt_10 (float): The percentage of relative errors less than 10%.
    - perc_error_lt_5 (float): The percentage of relative errors less than 5%.
    """

    # Test set
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    test_max_error = analysis_df_test['Rel_error_%'].max()
    test_median_error = analysis_df_test['Rel_error_%'].median()
    test_mean_error = analysis_df_test['+-Rel_error_%'].mean()
    test_mean_error_abs = analysis_df_test['Rel_error_%'].mean()

    test_perc_error_lt_10 = len(analysis_df_test[analysis_df_test['Rel_error_%'] < 10]) / len(analysis_df_test) * 100 # lt = less than
    test_perc_error_lt_5 = len(analysis_df_test[analysis_df_test['Rel_error_%'] < 5]) / len(analysis_df_test) * 100

    # Train set
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

    train_max_error = analysis_df_train['Rel_error_%'].max()
    train_median_error = analysis_df_train['Rel_error_%'].median()
    train_mean_error = analysis_df_train['+-Rel_error_%'].mean()
    train_mean_error_abs = analysis_df_train['Rel_error_%'].mean()

    perc_error_lt_10_train = len(analysis_df_train[analysis_df_train['Rel_error_%'] < 10]) / len(analysis_df_train) * 100 # lt = less than
    perc_error_lt_5_train = len(analysis_df_train[analysis_df_train['Rel_error_%'] < 5]) / len(analysis_df_train) * 100

    return {'r2_test': r2_test,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'mape_test': mape_test,
            'max_error_test': test_max_error,
            'median_error_test': test_median_error,
            'mean_error_test': test_mean_error,
            'mean_error_abs_test': test_mean_error_abs,
            'perc_error_lt_10_test': test_perc_error_lt_10,
            'perc_error_lt_5_test': test_perc_error_lt_5,
            'analysis_df_test': analysis_df_test,
            'r2_train': r2_train,
            'rmse_train': rmse_train,
            'mae_train': mae_train,
            'mape_train': mape_train,
            'max_error_train': train_max_error,
            'median_error_train': train_median_error,
            'mean_error_train': train_mean_error,
            'mean_error_abs_train': train_mean_error_abs,
            'perc_error_lt_10_train': perc_error_lt_10_train,
            'perc_error_lt_5_train': perc_error_lt_5_train,
            'analysis_df_train': analysis_df_train}



def return_final_track_dict(track_dict_list: list[dict[str, float]]) -> dict[str, float]:
    """
    Calculate and return a dictionary of final scoring metrics for a list of test sets.

    Args:
    -----
    - test_track_dict_list (list[dict[str, float]]): A list of dictionaries containing scoring metrics for each test set.

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
    r2_scores_test = []
    rmse_scores_test = []
    mae_scores_test = []
    mape_scores_test = []
    max_errors_test = []
    median_errors_test = []
    mean_errors_test = []
    mean_errors_abs_test = []
    lt_5_test = []
    lt_10_test = []
    final_analysis_df_test = pd.DataFrame()

    r2_scores_train = []
    rmse_scores_train = []
    mae_scores_train = []
    mape_scores_train = []
    max_errors_train = []
    median_errors_train = []
    mean_errors_train = []
    mean_errors_abs_train = []
    lt_5_train = []
    lt_10_train = []
    final_analysis_df_train = pd.DataFrame()

    for track_dict in track_dict_list:
        r2_scores_test.append(track_dict['r2_test'])
        rmse_scores_test.append(track_dict['rmse_test'])
        mae_scores_test.append(track_dict['mae_test'])
        mape_scores_test.append(track_dict['mape_test'])
        max_errors_test.append(track_dict['max_error_test'])
        median_errors_test.append(track_dict['median_error_test'])
        mean_errors_test.append(track_dict['mean_error_test'])
        mean_errors_abs_test.append(track_dict['mean_error_abs_test'])
        lt_5_test.append(track_dict['perc_error_lt_5_test'])
        lt_10_test.append(track_dict['perc_error_lt_10_test'])
        final_analysis_df_test = pd.concat([final_analysis_df_test, track_dict['analysis_df_test']])

        r2_scores_train.append(track_dict['r2_train'])
        rmse_scores_train.append(track_dict['rmse_train'])
        mae_scores_train.append(track_dict['mae_train'])
        mape_scores_train.append(track_dict['mape_train'])
        max_errors_train.append(track_dict['max_error_train'])
        median_errors_train.append(track_dict['median_error_train'])
        mean_errors_train.append(track_dict['mean_error_train'])
        mean_errors_abs_train.append(track_dict['mean_error_abs_train'])
        lt_5_train.append(track_dict['perc_error_lt_5_train'])
        lt_10_train.append(track_dict['perc_error_lt_10_train'])
        final_analysis_df_train = pd.concat([final_analysis_df_train, track_dict['analysis_df_train']])

    return {'R2 mean test': np.round(np.mean(r2_scores_test), 5),
            'RMSE mean test': np.round(np.mean(rmse_scores_test), 5),
            'MAE mean test': np.round(np.mean(mae_scores_test), 5),
            'MAPE mean test': np.round(np.mean(mape_scores_test), 5),
            'Max error % test': np.round(max(max_errors_test), 5),
            'Mean median error % test': np.round(np.mean(median_errors_test), 5),
            #'Mean error %': np.round(np.mean(mean_errors_test), 5),
            #'Mean abs error %': np.round(np.mean(mean_errors_abs_test), 5),
            'Mean % < 5% error test': np.round(np.mean(lt_5_test), 5),
            'Mean % < 10% error test': np.round(np.mean(lt_10_test), 5),
            'final_analysis_df_test': final_analysis_df_test,
            'R2 mean train': np.round(np.mean(r2_scores_train), 5),
            'RMSE mean train': np.round(np.mean(rmse_scores_train), 5),
            'MAE mean train': np.round(np.mean(mae_scores_train), 5),
            'MAPE mean train': np.round(np.mean(mape_scores_train), 5),
            'Max error % train': np.round(max(max_errors_train), 5),
            'Mean median error % train': np.round(np.mean(median_errors_train), 5),
            #'Mean error %': np.round(np.mean(mean_errors_train), 5),
            #'Mean abs error %': np.round(np.mean(mean_errors_abs_train), 5),
            'Mean % < 5% error train': np.round(np.mean(lt_5_train), 5),
            'Mean % < 10% error train': np.round(np.mean(lt_10_train), 5),
            'final_analysis_df_train': final_analysis_df_train
            }



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
    keys = ['target', 'model', 'features', 'feature_engineering', 'metric', 'random_state', 'n_iter', 'groupby_col', 'tune_hyps']
    return {key: parameters[key] for key in keys}



def wandb_final_results(parameters: dict,
                        final_score_dict: dict,
                        feature_importances_df: pd.DataFrame,
                        hyp_df: pd.DataFrame,
                        feature_importances_final_df: pd.DataFrame):
    """
    Track and log the final results of the model.

    Args:
    -----
    - parameters (dict): conf/base/parameters.yml parameter file.
    - final_score_dict (dict): Dictionary of scoring metrics.
    - feature_importances_df (pd.DataFrame): DataFrame containing the feature importances of the model.
    - hyp_df (pd.DataFrame): DataFrame containing the used hyperparameters in each fold.
    - feature_importances_final_df (pd.DataFrame): DataFrame containing the feature importances in each fold.

    Returns:
    --------
    - None. The function logs the results directly to the W&B dashboard.
    """
    experiment_name = _return_experiment_name(parameters)

    run = wandb.init(project=parameters['project_name'],
                     name=experiment_name,
                     notes=parameters['notes'],
                     tags=parameters['tags'],
                     config={**_return_config(parameters)} 
                     )
    
    # Log analysis dfs
    final_analysis_df_test = final_score_dict['final_analysis_df_test']
    wandb.log({'final_analysis_df_test': wandb.Table(dataframe=final_analysis_df_test.reset_index(), 
                                                allow_mixed_types=True)})
    
    final_analysis_df_train = final_score_dict['final_analysis_df_train']
    wandb.log({'final_analysis_df_train': wandb.Table(dataframe=final_analysis_df_train.reset_index(), 
                                                allow_mixed_types=True)})
    
    # Log hyperparameter df
    wandb.log({'hyp_df': wandb.Table(dataframe=hyp_df, allow_mixed_types=True)})

    # Log feature importances df
    wandb.log({'feature_importances_df': wandb.Table(dataframe=feature_importances_final_df, allow_mixed_types=True)})

    # List of keys to remove from the dictionary before logging
    keys_to_remove = ['final_analysis_df_train', 'final_analysis_df_test']

    wandb.log({k: v for k, v in final_score_dict.items() if k not in keys_to_remove})

    plot_predictions(final_analysis_df_test, parameters)
    plot_feature_importances(feature_importances_df, parameters)

    run.finish()



def plot_predictions(analysis_df: pd.DataFrame, 
                     parameters:dict, 
                     figsize: int=20,
                     zoom_into: int=5000):
    "Plot the predictions of the model across the k-fold test sets."

    fig, ax = plt.subplots(figsize=(figsize, figsize))

    CM_pred_col = [col for col in analysis_df.columns if col.endswith('_pred') if col.startswith('CM_')][0]
    CM_pred_col_idx = analysis_df.columns.get_loc(CM_pred_col)
    CM_real_col = analysis_df.columns[CM_pred_col_idx - 1]

    max_CM = max(analysis_df[CM_real_col].max(), analysis_df[CM_pred_col].max())
    max_lim = max_CM + max_CM/20

    ax.plot([0, max_lim], [0, max_lim], 'k--', label='Perfect predictions') # plot the 1:1 line

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*']  # Different markers for each test split

    for k_fold in analysis_df['k_fold_test'].unique():
        df_fold = analysis_df.loc[analysis_df['k_fold_test'] == k_fold, :]
        CM_real = df_fold[CM_real_col]
        CM_pred = df_fold[CM_pred_col]
        if k_fold == 1:
            fold_string = f'${k_fold}^{{st}}$ fold'
        elif k_fold == 2:
            fold_string = f'${k_fold}^{{nd}}$ fold'
        elif k_fold == 3:
            fold_string = f'${k_fold}^{{rd}}$ fold'
        else:
            fold_string = f'${k_fold}^{{th}}$ fold'

        ax.scatter(CM_real, CM_pred, label=f'{fold_string}, $R^2$={r2_score(CM_real, CM_pred):.4f}',
                   marker=markers[(k_fold-1) % len(markers)], s=figsize*10)

    # Adding zoomed in subplot with adjusted position
    axins = ax.inset_axes([0.525, 0.05, 0.425, 0.425])  # Adjusted for better positioning
    axins.plot([0, max_lim], [0, max_lim], 'k--')
    for k_fold in analysis_df['k_fold_test'].unique():
        df_fold = analysis_df.loc[analysis_df['k_fold_test'] == k_fold, :]
        CM_real = df_fold[CM_real_col]
        CM_pred = df_fold[CM_pred_col]
        axins.scatter(CM_real, CM_pred,
                      marker=markers[(k_fold-1) % len(markers)], s=figsize*10)

    axins.set_xlim(0, zoom_into)
    axins.set_ylim(0, zoom_into)
    axins.tick_params(axis='both', which='both', length=0)  # Remove ticks
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.grid()

    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2) #linestyle="--"
    plt.draw()

    # Function to format tick labels with commas
    def format_ticks(x, pos):
        return f'{x:,.0f}'

    target_dict = {'CM_e0': 'CM_1', 'CM_e1': 'CM_10', 'CM_e2': 'CM_100', 'CM_e3': 'CM_1,000', 'CM_e4': 'CM_10,000', 'CM_e5': 'CM_100,000'}
    CM_str = target_dict[CM_real_col]

    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax.set_xlabel(f'Actual {CM_str} [MPa]', fontsize=figsize*1.3, labelpad=figsize)
    ax.set_ylabel(f'Predicted {CM_str} [MPa]', fontsize=figsize*1.3, labelpad=figsize)
    ax.legend(fontsize=figsize*1.2, markerscale=2)
    ax.tick_params(axis='both', labelsize=figsize)
    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    ax.grid()

    if not parameters['plot_title']:
        ax.set_title(f'{CM_str} predictions', fontsize=figsize*1.6, pad=figsize*1.5)
        if parameters['use_wandb']:
            wandb.log({'test_predictions': wandb.Image('test_predictions.png')})
        else:
            plt.savefig('test_predictions.jpg', dpi=300, bbox_inches='tight')
    else:
        plot_title = parameters['plot_title']
        #ax.set_title(plot_title, fontsize=figsize*1.6, pad=figsize*1.5)
        if parameters['use_wandb']:
            wandb.log({'test_predictions': wandb.Image('test_predictions.png')})

        else:
            plt.savefig(f'{plot_title}.jpg', dpi=300, bbox_inches='tight')



def plot_feature_importances(feature_importances_df: pd.DataFrame, 
                             parameters: dict,
                             n_size: int=3):
    """Plot the feature importances of the model.
    """
    n_features = feature_importances_df.shape[0]
    plt.figure(figsize=(n_features*n_size*0.8, n_size*2))
    sns.boxplot(data=feature_importances_df.T)
    plt.title('Feature Importances Boxplot', fontsize=n_size*4.5)
    plt.ylabel('Normalized Importance', fontsize=n_size*4)
    plt.xlabel('Feature', fontsize=n_size*4)
    plt.xticks(fontsize=n_size*3.5)
    plt.yticks(fontsize=n_size*3.5)
    plt.grid()
    plt.ylim(0, 1)

    if parameters['use_wandb']:
        plt.savefig('feature_importances.png')  # Save the figure as an image file
        wandb.log({'feature_importances': wandb.Image('feature_importances.png')})  # Log the image file using wandb.Image()

    else:
        plt.savefig('feature_importances.png')



def validate_parameters(parameters: dict) -> None:
    """Raises ValueError: If any of the checks for conf/base/parameters.yml fails.
    """   
    # Check 'use_wandb' parameter
    if not isinstance(parameters['use_wandb'], bool):
        raise ValueError("use_wandb must be either True or False")

    # Check 'tune_hyps' parameter
    if not isinstance(parameters['tune_hyps'], bool):
        raise ValueError("tune_hyps must be either True or False")

    # Check hyperparameters
    if 'hyperparameters' in parameters:
        model_hyperparameters = parameters['hyperparameters'].get(parameters['model'])
        if model_hyperparameters is not None:
            for param, value in model_hyperparameters.items():
                if parameters['tune_hyps']:  # hyps. must be in a list
                    if not isinstance(value, list):
                        raise ValueError(f"Hyperparameter '{param}' must be in a list when tuning hyperparameters.")
                else:  # hyps. must be a single value, or None
                    if value is not None and isinstance(value, list):
                        raise ValueError(f"Hyperparameter '{param}' must be a single value or None when tune_hyps is False.")
    else:
        raise ValueError("No hyperparameters found in parameters.yml file.")