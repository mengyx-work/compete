import os, sys, time
import pandas as pd
import yaml
import cPickle as pickle

#sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.models import RandomForestModel, ExtraTreeModel
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params
from utils.bosch_functions import load_processed_bosch_data

models_yaml_file = 'raw_combined_models.yml'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
#project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'

with open(os.path.join(project_path, models_yaml_file), 'r') as yml_stream:
    models_dict = yaml.load(yml_stream)

dep_var_name = 'Response'
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file = 'bosch_processed_data_dict.yml'

train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0')

for index, model_dict in models_dict.items() :
    tmp_train = train.copy()
    train_label = train[dep_var_name]
    tmp_train.drop(dep_var_name, axis=1, inplace=True)

    if model_dict['model_type'] == 'ExtraTree':
        model = ExtraTreeModel(model_dict['model_params'])

    model.fit(tmp_train, train_label)
    model_pickle_file = 'indexed_{}_{}_model.pkl'.format(index, model_dict['model_type'])
    pickle.dump(model, open(model_pickle_file, 'wb'), -1)
    model_dict['model_file'] = model_pickle_file


trained_model_yaml_file = 'trained_combined_model.yml'
with open(os.path.join(project_path, trained_model_yaml_file), 'w') as yml_stream:
    yaml.dump(models_dict, yml_stream)
