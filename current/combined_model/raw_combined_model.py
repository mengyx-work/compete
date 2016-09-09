import os, sys, time
import yaml

project_path = '/home/ymm/kaggle/compete/current'
models_yaml_file = 'raw_combined_models.yml'
project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'

models = {}
model_index = 0
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 0, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict

model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 100, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict

model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 1000, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}

with open(os.path.join(project_path, models_yaml_file), 'w') as yml_stream:
    yaml.dump(models, yml_stream)
