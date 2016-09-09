import pandas as pd
import numpy as np
import os, sys, time
import yaml
from sklearn.metrics import matthews_corrcoef

## linux path
#sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
## Mac path
sys.path.append('/Users/ymm/Google_Drive/dev/xgboost_hyperopt')

import utils
from utils.wrapped_xgboost import xgboost_classifier
from utils.validation_tools import cross_validate_model, score_MCC



dep_var_name = 'Response'

'''
project_path = '/home/ymm/kaggle/compete/current'
data_path = '/home/ymm/kaggle/bosch_data/bosch_processed_data'

yaml_file = 'bosch_processed_data_dict.yml'
with open(os.path.join(project_path, yaml_file), 'r') as yml_stream:
    data_dict = yaml.load(yml_stream)

data_index = '0'
data_file = os.path.join(data_path, data_dict[data_index]['train_file'])
print 'loading data from ', data_file
'''
data_file = '/Users/ymm/Downloads/kaggle/bosch/processed_totBins_16_bin_0_train.csv'
train = pd.read_csv(data_file, index_col='Id')

params = {}
params["eta"]                      = 0.01
params["subsample"]                = 0.8
params["colsample_bytree"]         = 0.8
params["num_round"]                = 5000
params["max_depth"]                = 5
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 100

params["val"]                      = False
params["early_stopping_ratio"]     = 0.2

xgb_clf = xgboost_classifier(label_name = dep_var_name, params = params)
results = xgb_clf.cross_validate_fit(score_MCC, train, n_folds=2)
print results
