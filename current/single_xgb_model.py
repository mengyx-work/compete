import pandas as pd
import numpy as np
import os, sys, time

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.wrapped_xgboost import xgboost_classifier
from utils.models import RandomForestModel, ExtraTreeModel, XgboostModel
from utils.validation_tools import score_MCC, create_validation_index
from utils.bosch_functions import load_processed_bosch_data

dep_var_name     = 'Response'
'''
## smaller data
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path        = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file   = 'bosch_processed_data_dict.yml'
'''

project_yml_path = '/home/ymm/kaggle/compete/current/model_6_bins_data'
data_path = '/home/ymm/kaggle/bosch_data/bosch_complete_processed_6_bins_data'
data_yaml_file = 'complete_subset_data_6_bins_dict.yml'
#'''

train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='NaN')
print train.shape
train_index,valid_index = create_validation_index(train, 0.4, dep_var_name, True)  
valid_data  = train.ix[valid_index]
train       = train.ix[train_index]
valid_label = valid_data[dep_var_name]

params = {}
params["eta"]                      = 0.001
params["subsample"]                = 1.0
params["colsample_bytree"]         = 1.0
params["num_round"]                = 4500
params["max_depth"]                = 7
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 100
params['scale_pos_weight']         = 10.
## whether to use weights
params['use_weights']              = True

params["val"]                      = False
#params["early_stopping_ratio"]     = 0.2

#model = XgboostModel(params)
model = xgboost_classifier(label_name = dep_var_name, params = params, model_file='large_data_bin_NaN_xgb_model')
model.fit(train, dep_var_name)
valid_result = model.predict(valid_data)
print 'the MCC score:', score_MCC(valid_label, valid_result)

'''
## use the cross_validate_fit function
xgb_clf = xgboost_classifier(label_name = dep_var_name, params = params)
results = xgb_clf.cross_validate_fit(score_MCC, train, n_folds=2)
print 'results from cross validation:', results
'''
