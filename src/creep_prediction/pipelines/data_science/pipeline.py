from kedro.pipeline import Pipeline, node, pipeline

from .nodes import nested_cv, predictions_on_target

def create_pipeline(**kwargs) -> Pipeline:
    
    return pipeline(
        [
            node(func=nested_cv,
                inputs=["engineered_model_data", "info_data", "parameters"],
                outputs=["best_final_model", "model_evaluation_metrics"],
                name="nested_cv_node",
            ),
        ]
    )
    ''' 
    return pipeline(
        [
            node(func=predictions_on_target,
                inputs=["engineered_model_data", "info_data", "parameters"],
                outputs=None,#["target_data", "other_data"],
                name="return_target_df_node",
            ),
        ]
    )
    '''