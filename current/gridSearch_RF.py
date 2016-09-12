import pandas as pd
import numpy as np
import os, sys, time
import yaml

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.models import RandomForestModel, ExtraTreeModel
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params
from utils.bosch_functions import load_processed_bosch_data

## 15 bins data
#project_path = '/mnt/home/ymm/kaggle/compete/current'
#data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
#yaml_file = 'bosch_processed_data_dict.yml'

## 6 bins data
project_path = '/home/ymm/kaggle/compete/current/model_6_bins_data'
data_path = '/home/ymm/kaggle/bosch_data/bosch_complete_processed_6_bins_data'
data_yaml_file = 'complete_subset_data_6_bins_dict.yml'
data_index= '0'
train = load_processed_bosch_data(data_path, project_path, data_yaml_file, data_index=data_index)

const_param_dict = {'random_state' : 0, 'criterion' : 'entropy', 'n_jobs' : -1}
#const_param_dict = {'criterion' : 'entropy', 'n_jobs' : -1, 'n_estimators' : 500}
const_param_dict = list_const_params(const_param_dict)
#tuning_param_dict = {'random_state': range(0, 10), 'max_depth':[4, 7]}
tuning_param_dict = {'n_estimators': [500, 1000, 1500, 2000, 2500], 'max_depth':range(3, 8)}

param_dict = const_param_dict.copy()
param_dict.update(tuning_param_dict)

cv_fold_num = 2
result_file_name = 'RandomForest_{}_{}_fold_GridSearch_Results.csv'.format(const_param_dict['criterion'][0], cv_fold_num)
grid_search_cross_validate_model(train, dep_var_name, RandomForestModel, score_MCC, param_dict, cv_fold_num, result_file=result_file_name)


'''
project_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'

yaml_file = 'bosch_processed_data_dict.yml'
with open(os.path.join(project_path, yaml_file), 'r') as yml_stream:
    data_dict = yaml.load(yml_stream)

dep_var_name = 'Response'
data_index= '0'
data_file = os.path.join(data_path, data_dict[data_index]['train_file'])
print 'loading data from ', data_file
train = pd.read_csv(data_file, index_col='Id')

const_param_dict = {'random_state' : 0, 'criterion' : 'gini', 'n_jobs' : -1, 'max_depth' : 3}
const_param_dict = list_const_params(const_param_dict)
#tuning_param_dict = {'n_estimators':range(400, 4000, 200), 'max_depth':range(4, 8)}
tuning_param_dict = {'n_estimators' : range(300, 2000, 200)}

param_dict = const_param_dict.copy()
param_dict.update(tuning_param_dict)

grid_search_cross_validate_model(train, dep_var_name, RandomForestModel, score_MCC, param_dict, 2, result_file='RF_gini_3_depth_GridSearch_2fold_Results.csv')
'''
