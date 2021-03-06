import os, sys, time
import pandas as pd
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.bosch_functions import load_processed_bosch_data
from utils.models import CombinedModel
from utils.validation_tools import score_MCC, create_validation_index

## params for the combined model
dep_var_name = 'Response'
raw_models_yaml_file = 'raw_combined_models.yml'
trained_model_yaml_file = 'trained_combined_model.yml'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
trained_model_yaml_file = 'trained_combined_model.yml'

## load training data
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file = 'bosch_processed_data_dict.yml'
train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0')
train_index,valid_index = create_validation_index(train, 0.2, dep_var_name, True)  
valid_data = train.ix[valid_index]
train      = train.ix[train_index]

## train the comined model
combined_model_params = {}
combined_model_params['raw_models_yaml_file'] = raw_models_yaml_file
combined_model_params['project_path'] = project_path
combined_model_params['models_yaml_file'] = trained_model_yaml_file

combined_model = CombinedModel(combined_model_params)
combined_model.fit(train, dep_var_name)
pred_df = combined_model.predict(valid_data)
pred_df.to_csv('tmp.csv')


