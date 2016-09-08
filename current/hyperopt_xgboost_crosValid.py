import pandas as pd
import os, sys, time
import yaml


#HOME_DIRECTORY = os.path.expanduser('~')
#PROJECT_UTILS_PATH = HOME_DIRECTORY+'/Google Drive/dev/xgboost_hyperopt_wrapper/xgboost_hyperopt/utils'
#sys.path.append(PROJECT_UTILS_PATH)
sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')


# suppress various warnings
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from utils.hyperopt_xgboost import hyperopt_xgboost
from utils.utils_functions import utils_functions

# the parameters to tune by hyperopt
# parameter can be
# a. three element tuple
# b. a list of discrete values
tuning_params = {'eta':(0.001, 0.01, 0.0005), 'num_round':(1000, 10000, 500)}

# constant parameters as a dictionary
# avant standard xgboost params
params = {}
params["eta"]                      = 0.0075
params["subsample"]                = 0.8
params["colsample_bytree"]         = 0.8
params["num_round"]                = 10
params["max_depth"]                = 5
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 100

params["val"]                      = False ## whether create a valid dataset
params["early_stopping_ratio"]     = 0.2

dep_var_name = 'Response'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_processed_data'
project_path = '/mnt/home/ymm/kaggle/compete/current'

yaml_file = 'bosch_processed_data_dict.yml'
with open(os.path.join(project_path, yaml_file), 'r') as yml_stream:
        data_dict = yaml.load(yml_stream)

data_index = '0'
data_file = os.path.join(data_path, data_dict[data_index]['train_file'])
print 'loading data from ', data_file
data = pd.read_csv(data_file, index_col='Id')


hyperopt = hyperopt_xgboost(data, dep_var_name,
                            tuning_params = tuning_params,
                            const_params  = params,
                            data_filename = 'bosch_xgboost_hyperopt_results.csv',
                            crosValid_mode = True,
                            fold_num = 3,
                            max_opt_count = 400)
hyperopt.hyperopt_run()
