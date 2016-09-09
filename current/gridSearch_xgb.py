import pandas as pd
import numpy as np
import os, sys, time
import yaml

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
#from utils.models import RandomForestModel, ExtraTreeModel
from utils.wrapped_xgboost import xgboost_classifier
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params, combine_tuning_params


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

const_param_dict = {}
const_param_dict["eta"]                      = 0.001
#const_param_dict["subsample"]                = 0.8
#const_param_dict["colsample_bytree"]         = 0.8
const_param_dict["num_round"]                = 10000
const_param_dict["max_depth"]                = 5
const_param_dict["gamma"]                    = 0
const_param_dict["metrics"]                  = 'auc'
const_param_dict['eval_metric']              = 'auc'
const_param_dict["seed"]                     = 100

const_param_dict["val"]                      = False
#const_param_dict["early_stopping_ratio"]     = 0.2

const_param_dict = list_const_params(const_param_dict)
tuning_param_dict = {'num_round': [0.0075, 0.005], 'num_round' : [2500, 3500, 4500, 5500, 6500], 'max_depth' : [3, 4, 5, 6, 7]}
param_dict = combine_tuning_params(const_param_dict, tuning_param_dict)

grid_search_cross_validate_model(train, dep_var_name, xgboost_classifier, score_MCC, param_dict, 2, result_file='xgb_GridSearch_2fold_Results.csv', is_xgb_model=True)

