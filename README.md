# Polymer creep modulus prediction by leveraging CAMPUS database and gradient boosting

## How to install dependencies

Create conda environment:

```
conda create -n ENV_NAME python=3.10.9
```
And install requirements:

```
pip install -r src/requirements.txt
```

Or directly from the environment.yml file (slower):

```
conda env create -f src/environment.yml
```

Then activate the environment and you are ready to go.


## MLOps tools:

This project uses [Kedro](https://kedro.readthedocs.io/en/stable/) for project structuring and data pipelines, and [Weights & Biases](https://docs.wandb.ai/) for experiment tracking. Refer to the documentation of these tools for more information.

## How to run Kedro pipelines:

You can run all the pipelines in `src/creep_prediction/pipelines` with:

```
kedro run
```
Or run a specific pipeline with:

```
kedro run --pipeline PIPELINE_NAME
```

## How to define experiments:

All the necessary parameters for the experiments are defined in `conf/base/parameters.yml`. Additional information about the parameters can be found in the same file.


## Experiment tracking:

To track the experiments Weights & Biases, you need to create a account and set `use_wandb: True` in `conf/base/parameters.yml`. If set to `False`, you can inspect the `model_evalution_metrics.pickle` in `data/05_reporting`.