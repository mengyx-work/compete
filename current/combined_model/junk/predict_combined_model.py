import pandas as pd
import os, sys, time
import yaml
import cPickle as pickle

#sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.models import RandomForestModel, ExtraTreeModel
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params, create_validation_index
from utils.bosch_functions import load_processed_bosch_data


models_yaml_file = 'trained_combined_model.yml'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
#project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'

with open(os.path.join(project_path, models_yaml_file), 'r') as yml_stream:
    models_dict = yaml.load(yml_stream)

dep_var_name = 'Response'
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file = 'bosch_processed_data_dict.yml'

train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0')
train_index,valid_index = create_validation_index(train, 0.2, dep_var_name, True)  
tmp_train = train.ix[train_index]
valid_data = train.ix[valid_index]
tmp_train_label = tmp_train[dep_var_name]
valid_label = valid_data[dep_var_name]
tmp_train.drop(dep_var_name, axis=0, inplace=True)
valid_data.drop(dep_var_name, axis=0, inplace=True)
print train.shape,tmp_train.shape

for index, model_dict in models_dict.items() :
    model_pickle_file = model_dict['model_file']
    model = pickle.load(open(model_pickle_file, 'rb'))
    res = model.predict(valid_data)
    print score_MCC(valid_label, res)


