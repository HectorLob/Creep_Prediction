from ruamel.yaml import YAML
import subprocess
import shutil
import os
import itertools

# Create backup directory if it doesn't exist
if not os.path.exists('conf/backup/'):
    os.makedirs('conf/backup/')

# Make a backup copy of the original parameters.yml file
shutil.copy('conf/base/parameters.yml', 'conf/backup/parameters_backup.yml')

models = ['LinearRegression',
          'DecisionTreeRegressor', 'RandomForestRegressor',
          'LGBMRegressor', 'XGBRegressor']

model_name_dict = {'LinearRegression': 'LR',
                   'DecisionTreeRegressor': 'DT',
                   'RandomForestRegressor': 'RF',
                   'LGBMRegressor': 'LGBM',
                   'XGBRegressor': 'XGB'}

exp_dict = {'MC1': {'target': 'R_e3/e2',
                   'features': ['R_e2/e1', 'R_e1/e0']},
            'MC2': {'target': 'R_e4/e3',
                    'features': ['R_e3/e2', 'R_e2/e1', 'R_e1/e0']},
            'MC3': {'target': 'R_e4/e2',
                    'features': ['R_e2/e1', 'R_e1/e0']},              
}

yaml = YAML()
yaml.preserve_quotes = True
    
# Generate all possible combinations
combinations = list(itertools.product(models, exp_dict.items()))

# Iterate over all combinations
for i, (model, (exp_name, exp_config)) in enumerate(combinations, start=1):

    print(f"Running combination {i}/{len(combinations)}:")
    print(f"model = {model}")
    print(f"target = {exp_config['target']}")
    print(f"features = {exp_config['features']}")
    # Load parameters
    with open('conf/base/parameters.yml', 'r') as file:
        parameters = yaml.load(file)

    # Update parameters
    parameters['model'] = model
    parameters['target'] = exp_config['target']
    parameters['features'] = exp_config['features']
    parameters['tags'] = [exp_name]

    exp_title = model_name_dict[model] + '_' + exp_name 
    parameters['plot_title'] = exp_title
    parameters['experiment_name'] = exp_title
    #parameters['hyperparameters'][model] = {}

    # Save parameters
    with open('conf/base/parameters.yml', 'w') as file:
        yaml.dump(parameters, file)

    # Run kedro
    subprocess.run(['kedro', 'run'])

    # Restore the original parameters.yml file from the backup
    shutil.copy('conf/backup/parameters_backup.yml', 'conf/base/parameters.yml')