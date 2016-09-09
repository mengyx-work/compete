import pandas as pd
import os, sys, time
import yaml
import cPickle as pickle

#sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.models import RandomForestModel, ExtraTreeModel
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params


models_yaml_file = 'trained_combined_model.yml'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
#project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'

with open(os.path.join(project_path, models_yaml_file), 'r') as yml_stream:
    models_dict = yaml.load(yml_stream)


dep_var_name = 'Response'
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file = 'bosch_processed_data_dict.yml'

def load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0'):
    with open(os.path.join(project_yml_path, data_yaml_file), 'r') as yml_stream:
        data_dict = yaml.load(yml_stream)

    data_file = os.path.join(data_path, data_dict[data_index]['train_file'])
    print 'loading bosch data from ', data_file
    train = pd.read_csv(data_file, index_col='Id', nrows=1000)
    return train

train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0')


for index, model_dict in models_dict.items() :

    model_pickle_file = model_dict['model_file']
    model = pickle.load(open(model_pickle_file, 'rb'))
    res = model.predict(train)
    print res[0:10]


