from kedro.pipeline import Pipeline, node, pipeline

from .nodes import nested_cv

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