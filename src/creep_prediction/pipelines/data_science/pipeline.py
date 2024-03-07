from kedro.pipeline import Pipeline, node, pipeline

from .nodes import main # to use unscaled features, import .nodes_unscaled


def create_pipeline(func=main, **kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=func,
                inputs=["engineered_model_data", "info_data", "parameters"],
                outputs=["best_final_model", "model_evaluation_metrics"],
                name="cv_node",
            ),
        ]
    )