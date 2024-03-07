import pandas as pd
import numpy as np



def _get_features(parameters: dict) -> list[str]:
    """Get the features from the parameters dict"""
    features = parameters.get('features', []) # Get the features from the parameters dict or return an empty list if not present
    
    if isinstance(features, str): # Convert the string to a list if it is a string
        features = [features]

    if not features: # Check if the features list is empty
        raise ValueError("Some features should be specified in the parameters.yml file.")

    return features



def split_df(preprocessed_creep_data: pd.DataFrame, 
             parameters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data DataFrame into 2 DataFrames: model_data and info_data.

    Args:
    -----
    - preprocessed_creep_data: The data DataFrame.
    - parameters: The parameters from the parameters.yml file.

    Returns:
    --------
    - model_data: The df for modeling.
    - info_data: The info df.
    """
    df = preprocessed_creep_data.copy()

    # Retrieving parameters
    target = parameters['target']
    features = _get_features(parameters)
    family_col = parameters['groupby_col']

    # Checking for necessary columns in the DataFrame
    for column in [target, family_col] + features:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in data DataFrame.")
        
    # Constructing the model data DataFrame
    model_data = df[features + [family_col] + [target]].dropna(axis=0) # Drop rows with NaNs

    print(f"Number of rows dropped due to NaNs: {len(df) - len(model_data)} / {len(df)}")

    # Constructing the info data DataFrame
    info_cols = ['Material', 'Family', 'Material_info', 'Dry_or_cond', 'Manufacturer', 'Polymer', 'Fiber_type']
    CM_cols = [col for col in df.columns if col.startswith('CM_')]

    # Checking if ratios are used in target or features
    if target.startswith('R_') or any([f.startswith('R_') for f in features]): 
        R_cols = [col for col in df.columns if col.startswith('R_')]
    else: 
        R_cols = []

    # Creating the info_data df, with the same index as the model_data df
    info_data = df[info_cols + list(set(features + R_cols + CM_cols))].iloc[model_data.index]

    # Sorting the columns
    other_cols = [col for col in info_data.columns if col not in CM_cols + R_cols]
    info_data = info_data[other_cols + sorted(R_cols) + sorted(CM_cols)] 

    return model_data, info_data



def log_features(model_data: pd.DataFrame, 
                 parameters: dict) -> pd.DataFrame:
    """Create new columns with the log of the features.
    """
    features = _get_features(parameters)

    new_features = []

    # For every feature, create a new column with its log
    for feature in features:
        model_data[f"*{feature}_log"] = np.log(model_data[feature])
        new_features.append(f"{feature}_log")

    # Print created features
    print(f"Created log features: {new_features}")
    
    return model_data



def feature_interactions(model_data: pd.DataFrame, 
                         parameters: dict) -> pd.DataFrame:
    """Create new columns with the interactions between the features."""
    features = _get_features(parameters)

    new_features = []

    # Create new columns for feature interactions without repetition
    for i, feat1 in enumerate(features):
        for feat2 in features[i+1:]:  # Avoid duplicates by starting from the next feature
            interaction_col_name = f"{feat1}*{feat2}"
            model_data[interaction_col_name] = model_data[feat1] * model_data[feat2]
            new_features.append(interaction_col_name)
        
    # Print created features
    print(f"Created interaction features: {new_features}")

    return model_data



def square_ratios(model_data: pd.DataFrame, 
                  parameters) -> pd.DataFrame:
    """Create new columns with the squared values of the ratios.
    parameters dict is not used, but it is needed for feature_engineering function to work.
    """
    R_cols = [col for col in model_data.columns if col.startswith('R_')]
    R_cols.remove(parameters['target'])

    new_features = []

    for col in R_cols:
        model_data[f"*{col}_squared"] = model_data[col]**2
        new_features.append(f"{col}_squared")

    # Print created features
    print(f"Created squared ratio features: {new_features}")

    return model_data



def feature_engineering(model_data: pd.DataFrame, 
                        parameters: dict) -> pd.DataFrame:
    """
    Apply a set of feature engineering techniques to the data.

    The techniques are specified in the parameters dict, and can either be a single technique (str), 
    a list of techniques, or blank/'null' to apply no transformations. 
    Each technique is a function that accepts a DataFrame as input and returns another one.
    At the end, if multiple techniques are applied, the DataFrames are concatenated.

    Args:
    -----
    - model_data (pd.DataFrame): The input DataFrame to be transformed. Note: The original df will not be modified.
    - parameters (dict): conf/base/parameters.yml parameter file

    Returns:
    -------
    - engineered_model_data (pd.DataFrame): The transformed DataFrame.

    Raises:
    ------
    - ValueError: If a technique specified in the parameters dict is not a valid function in the current namespace.
    """

    # Initialize an empty list to store transformed DataFrames
    transformed_dfs = []

    # Ensure techniques are in a list
    techniques = parameters['feature_engineering']

    # If no techniques are specified, return the original DataFrame
    if not techniques or not techniques[0]:
        print("No feature engineering techniques applied.")
        return model_data
    
    else:
        # If only one technique is specified, put it in a list
        if isinstance(techniques, str):
            techniques = [techniques]

        for technique in techniques:
            # Check if the technique is a function in the current or imported modules
            if technique in globals() and callable(globals()[technique]):
                print(f"Applying feature engineering technique: {technique}")
                # Apply the transformation and add the result to our list
                transformed_df = globals()[technique](model_data, parameters)
                transformed_dfs.append(transformed_df)
            else:
                raise ValueError(f"{technique} is not a valid feature engineering technique.")

        # Concatenate all the transformed DataFrames along the columns axis
        engineered_model_data = pd.concat(transformed_dfs, axis=1)

        # Remove duplicate columns
        engineered_model_data = engineered_model_data.loc[:,~engineered_model_data.columns.duplicated()]

        # Set target as last column
        target = parameters['target']
        engineered_model_data = engineered_model_data[[col for col in engineered_model_data.columns if col != target] + [target]]

        # Print the columns of the DataFrame
        print(f"Columns of the DataFrame after feature engineering: {engineered_model_data.columns}")

        return engineered_model_data