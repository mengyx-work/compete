import pandas as pd
import numpy as np
import os, sys, time
import yaml
from sklearn.metrics import matthews_corrcoef

project_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_processed_data'

yaml_file = 'bosch_processed_data_dict.yml'
with open(os.path.join(project_path, yaml_file), 'r') as yml_stream:
    data_dict = yaml.load(yml_stream)

data_index = '0'
data_file = os.path.join(data_path, data_dict[data_index]['train_file'])
print 'loading data from ', data_file
train = pd.read_csv(data_file, index_col='Id')
print train.head()

