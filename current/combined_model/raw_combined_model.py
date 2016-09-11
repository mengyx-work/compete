import os, sys, time
import yaml

project_path = '/home/ymm/kaggle/compete/current'
models_yaml_file = 'raw_combined_models.yml'
#project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'

models = {}

## for different random seed
def auto_fill_model_dicts(models, const_params, list_param_dict, model_type, model_index=None):
    if model_index is None:
        model_index = -1

    for key, list_param in list_param_dict.items():
        for param in list_param:
            model_params = const_params.copy()
            model_dict = {}
            model_index += 1
            model_params[key] = param
            model_dict['model_params'] = model_params
            model_dict['model_type'] = model_type
            print 'build a {} model using params: {}'.format(model_type, model_params)
            models[model_index] = model_dict


const_params = {'random_state' : 9999, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
list_param_dict = {}
#list_param_dict['random_state'] = range(0, 10000, 100)
list_param_dict['random_state'] = range(0, 3)

auto_fill_model_dicts(models, const_params, list_param_dict, 'ExtraTree')


with open(os.path.join(project_path, models_yaml_file), 'w') as yml_stream:
    yaml.dump(models, yml_stream, default_flow_style=False)

'''
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
models[model_index] = model_dict


model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 9999, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict
'''

