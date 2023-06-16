from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_creep_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_creep_data,
                inputs="creep_data", 
                outputs="preprocessed_creep_data",
                name="preprocess_creep_data_node",
            ),
        ]
    )