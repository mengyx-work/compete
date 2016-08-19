import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

data_path = '/home/ymm/data/talkingdata_data/'
event_file = 'events.csv'
agg_app_event_labeled_file = 'agg_app_event_labeled_data.csv'
agg_app_event_labeled = read_csv(data_path + agg_app_event_labeled_file, index_col='event_id')
print 'agg_app_event_labeled is loaded from file with shape:', agg_app_event_labeled.shape

events = pd.read_csv(data_path + event_file, index_col='event_id')
print 'the events shape:', events_labeled.shape
print '#unique device_id:', len(events_labeled.device_id.unique())
events_labeled = pd.merge(events, agg_app_event_labeled, how='left', left_index=True, right_index=True)

## convert datetime variable
events_labeled.timestamp = pd.to_datetime(events_labeled.timestamp)
## drop event_id as index
events_labeled = events_labeled.reset_index(drop=True)

## on device_id to aggregate, calculate the time_min, time_max and the time_diff
time_group_func_dict = {'device_time_max' : max, 'device_time_min' : min, 'device_time_diff' : lambda x: (max(x) - min(x))}
events_time_fea = events_labeled[['timestamp', 'device_id']].groupby('device_id').agg(time_group_func_dict.values())
## rename the column names
events_time_fea.columns = time_group_func_dict.keys()
print 'event level, time feature shape:', events_time_fea.shape

## the time_max and time_min is converted into seconds by reference to the minimal value of timestamp
ref_timepoint = tmp_events_labeled.timestamp.min()

def convert2seconds(x):
    return x.seconds

def ref_with_min_time(x):
    time_diff = x - ref_timepoint
    return time_diff.seconds

events_time_fea['device_time_min'] = events_time_fea.loc[:, 'device_time_min'].apply(ref_with_min_time)
events_time_fea['device_time_max'] = events_time_fea.loc[:, 'device_time_max'].apply(ref_with_min_time)
time_diff_var = 'device_time_diff'
events_time_fea[time_diff_var] = events_time_fea.loc[:, time_diff_var].apply(convert2seconds)


'''
regular feature aggregation section
'''

def remove_content(long_list, short_list):
    for elem in short_list:
        if elem in long_list:
            long_list.remove(elem)
    return long_list

fea2remove = ['timestamp', 'longitude', 'latitude', 'device_id']
fea_names = tmp_events_labeled.columns.tolist()
fea_list = remove_content(fea_names, fea2remove)

## generate location-related features from three raw features
fea_func_dict = {'max' : np.max, 'min' : np.min, 'mean' : np.mean, 'sum' : np.sum, 'size' : np.size}
fea_df = None
for feature_name in fea_list:
    ## derive features from single feature using functions from a dictionary called geo_fea_func_dict
    tmp_df = agg_column_derive_fea(tmp_events_labeled, 'device_id', feature_name, fea_func_dict)
    ## concatenate all the features together
    if fea_df is None:
        fea_df = tmp_df
    else:
        fea_df = pd.merge(fea_df, tmp_df, how='outer', left_index=True, right_index=True)


## merge regular feature data with the time features
print 'regular app_category features shape:', fea_df.shape, 'time feature shape:', events_time_fea.shape
device_feature_df = pd.merge(fea_df, events_time_fea, how='outer', left_index=True, right_index=True)
print 'after merging, device_feature shape:', device_feature_df.shape

## separate the location columns
location_events = events_labeled[['longitude', 'latitude', 'device_id']]
location_events['approx_dist'] = location_events[['longitude', 'latitude']].apply(lambda x: np.sqrt(x[0]*x[0] + x[1]*x[1]), axis = 1)


def agg_column_derive_fea(dataFrame, agg_key_name, column_name, fea_func_dict):
    derived_df = dataFrame[[agg_key_name, column_name]].groupby(agg_key_name).agg(fea_func_dict.values())
    derived_df.columns = [column_name + '_' + elem for elem in fea_func_dict.keys()]
    return derived_df

## generate location-related features from three raw features
geo_fea_func_dict = {'max' : np.max, 'min' : np.min, 'mean' : np.mean}
geo_fea_df = None
for feature_name in ['longitude', 'latitude', 'approx_dist']:
    ## derive features from single feature using functions from a dictionary called geo_fea_func_dict
    tmp_df = agg_column_derive_fea(tmp_events_labeled, 'device_id', feature_name, geo_fea_func_dict)
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
pd.to_csv(data_path + device_feature_file)


