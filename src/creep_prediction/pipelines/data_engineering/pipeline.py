from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_df, feature_engineering

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_df,
                inputs=["preprocessed_creep_data", "parameters"],
                outputs=["model_data", "info_data"],
                name="split_df_node",
            ),
            node(
                func=feature_engineering,
                inputs=["model_data", "parameters"],
                outputs="engineered_model_data",
                name="feature_engineering_node",
            ),
        ]
    )