import pandas as pd

def _rename_cols(creep_data: pd.DataFrame) -> pd.DataFrame:
    """Rename some columns
    """
    rename_dict = {'CM_1.0': 'CM_1',
                   'CM_10.0': 'CM_10',
                   'CM_100.0': 'CM_100',
                   'CM_1000.0': 'CM_1000',
                   'CM_10000.0': 'CM_10000',
                   'CM_100000.0': 'CM_100000',
                   'MPa': 'Load'}
    
    # If keys in dict are in the df, rename them
    return creep_data.rename(columns={key: rename_dict[key] for key in rename_dict.keys() if key in creep_data.columns})


def _create_ratio_cols(creep_data: pd.DataFrame) -> pd.DataFrame:
    """Create ratio columns for the creep_data DataFrame, where
    R_X/Y = CM_X / CM_Y, where X h > Y h.
    """  
    creep_data['R_10/1'] = creep_data['CM_10'] / creep_data['CM_1']
    creep_data['R_100/10'] = creep_data['CM_100'] / creep_data['CM_10']
    creep_data['R_1000/100'] = creep_data['CM_1000'] / creep_data['CM_100']
    creep_data['R_10000/1000'] = creep_data['CM_10000'] / creep_data['CM_1000']
    creep_data['R_100000/10000'] = creep_data['CM_100000'] / creep_data['CM_10000']

    return creep_data



def _delete_incorrect_rows(creep_data: pd.DataFrame) -> pd.DataFrame:
    """Delete rows where R_10/1 < 0.5, as these are incorrect.
    """
    drop_rows = creep_data[creep_data['R_10/1'] < 0.5].index

    return creep_data.drop(drop_rows, axis=0).reset_index(drop=True)



def _split_material_col(creep_data: pd.DataFrame) -> pd.DataFrame:
    """Split the material column into two columns, where 
    - 1st column contains the material name and 
    - 2nd column contains the material number.
    """
    creep_data['Family'] = creep_data['Material'].apply(lambda x: x.split(' ')[0])
    creep_data['Material_info'] = creep_data['Material'].apply(lambda x: ' '.join(x.split(' ')[1:]))

    # reorder 'Family' and 'Material_info' columns after 'Material'
    first_cols = ['Material', 'Family', 'Material_info']

    return creep_data[first_cols + [col for col in creep_data.columns if col not in first_cols]]



def preprocess_creep_data(creep_data: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the creep_data DataFrame.
    """
    creep_data = creep_data.copy()
    creep_data = _rename_cols(creep_data)
    creep_data = _create_ratio_cols(creep_data)
    creep_data = _delete_incorrect_rows(creep_data)
    creep_data = _split_material_col(creep_data)

    return creep_data