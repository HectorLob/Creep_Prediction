# Prediction of long-term creep modulus of thermoplastics by brief testing data and interpretable machine learning

This repository contains the code to predict the Creep Modulus of thermoplastics using public material data and machine learning. This research project, conducted at [Leartiker](https://www.leartiker.com/), has been accepted for publication in International Journal of Solids and Structures, doi: https://doi.org/10.1016/j.ijsolstr.2024.113014

All credit for the dataset to the [CAMPUS Plastics](https://www.campusplastics.com/) material information system. 

Corresponding author: Héctor Lobato (hlobato@leartiker.com) 


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

This repository uses [Kedro](https://kedro.readthedocs.io/en/stable/) for project structuring and data pipelines, and [Weights & Biases](https://docs.wandb.ai/) for experiment tracking. Refer to the documentation of these tools for more information.


## Experiment tracking:

To track the experiments with Weights & Biases, you need to create an account and set `use_wandb: True` in `conf/base/parameters.yml`. If set to `False`, you can inspect the `model_evalution_metrics.pickle` corresponding to the experiment in `data/05_reporting`.


## How to run Kedro pipelines:

You can run all the pipelines in `src/creep_prediction/pipelines` with:

```
kedro run
```
Or run a specific pipeline with:

```
kedro run --pipeline PIPELINE_NAME
```

To visualize the project structure and data pipelines, run:

```
kedro viz
```

## How to define experiments:

All the necessary parameters for the experiments are defined in `conf/base/parameters.yml`. Additional information about the parameters can be found in the same file.


## How to replicate paper results:

Execute `run.py`
