import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

data_path = '/home/ymm/data/talkingdata_data/'
tmp_device_feature_file = 'events_time_fea_with_reg_fea_data.csv'
events_labeled_file = 'events_labeled_data.csv'
if not os.path.isfile(data_path + tmp_device_feature_file) or not os.path.isfile(data_path + events_labeled_file):
    sys.exit('failed to find file')

device_feature_df = pd.read_csv(data_path + tmp_device_feature_file, index_col='device_id')
print 'device_feature_df of time_fea and regular_fea are loaded with shape:', device_feature_df.shape

events_labeled = pd.read_csv(data_path + events_labeled_file, index_col='event_id')
print 'events_labeled is loaded with shape:', events_labeled.shape 


def agg_column_derive_fea(dataFrame, agg_key_name, column_name, fea_func_dict):
    derived_df = dataFrame[[agg_key_name, column_name]].groupby(agg_key_name).agg(fea_func_dict.values())
    derived_df.columns = [column_name + '_' + elem for elem in fea_func_dict.keys()]
    return derived_df


## separate the location columns
location_events = events_labeled[['longitude', 'latitude', 'device_id']]
location_events['approx_dist'] = location_events[['longitude', 'latitude']].apply(lambda x: np.sqrt(x[0]*x[0] + x[1]*x[1]), axis = 1)

## generate location-related features from three raw features
geo_fea_func_dict = {'max' : np.max, 'min' : np.min, 'mean' : np.mean}
geo_fea_df = None
for feature_name in ['longitude', 'latitude', 'approx_dist']:
    ## derive features from single feature using functions from a dictionary called geo_fea_func_dict
    tmp_df = agg_column_derive_fea(location_events, 'device_id', feature_name, geo_fea_func_dict)
    ## concatenate all the features together
    if geo_fea_df is None:
        geo_fea_df = tmp_df
    else:
        geo_fea_df = pd.merge(geo_fea_df, tmp_df, how='outer', left_index=True, right_index=True)
        
## merge location feature to the device_feature df
print 'location feature shape:', geo_fea_df.shape
device_feature_df = pd.merge(device_feature_df, geo_fea_df, how='outer', left_index=True, right_index=True)
print 'after merging with location feature shape:', device_feature_df.shape

device_feature_file = 'devide_feature_data.csv'
device_feature_df.to_csv(data_path + device_feature_file)


