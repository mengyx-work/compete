import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack


#data_path = '/Users/ymm/Downloads/kaggle/talkingdata_data/'
data_path = '/home/ymm/data/talkingdata_data/'

app_labels_file               = 'app_labels.csv'
app_event_file                = 'app_events.csv'
label_categories_file         = 'label_categories.csv'
event_file                    = 'events.csv'
gender_age_train_file         = 'gender_age_train.csv'
gender_age_test_file          = 'gender_age_test.csv'
phone_brand_device_model_file = 'phone_brand_device_model.csv'

## load the .csv files
app_labels = pd.read_csv(data_path + app_labels_file)
label_categories = pd.read_csv(data_path + label_categories_file)

'''
section to create app_label table from label_categories
'''

## section to creat the sparse matrix for app_label table
unique_app_ids = app_labels.app_id.unique()
app_id_df = pd.DataFrame({'app_id':unique_app_ids, 'rowIndx':range(len(unique_app_ids))})

## the app_id and rowIndx two columns already gives the one-to-one between app_id and the rowIndx
tmp_app_labels = app_labels.merge(app_id_df, on='app_id', how='left')

## since the label_id is not continous, transform it into continous integers so that it can be used as column number
app_label_encoder = LabelEncoder().fit(tmp_app_labels.label_id)
tmp_app_labels.label_id = app_label_encoder.transform(tmp_app_labels.label_id) 
tmp_csr_matrix = csr_matrix((np.ones(tmp_app_labels.shape[0]), (tmp_app_labels.rowIndx, tmp_app_labels.label_id)))
sparse_app_labels = pd.DataFrame(tmp_csr_matrix.toarray())

## new columnsn names are labeled by the orginal *label_id*
sparse_app_labels.columns = ["app_label_" + str(i) for i in app_label_encoder.classes_]
sparse_app_labels['app_id'] = app_id_df.app_id

processed_app_label_file = 'processed_app_label_table.csv'
sparse_app_labels.to_csv(data_path + processed_app_label_file, index=False)
print 'sparse_app_labels with shape:', sparse_app_labels.shape, 'is saved into file:', processed_app_label_file


'''
process the app_event and join it with newly created sparse_app_labels
'''


app_event = pd.read_csv(data_path + app_event_file, usecols= ['app_id', 'is_active', 'event_id'])


def update_app_event_column_name(column_names, level):
    corr_column_names = []
    for column_name in column_names:
        corr_column_names.append("active_" + str(level) + "_" + column_name)
    return corr_column_names
    
def split_aggregate_combine(split_column_name, agg_column_name, dataFrame):
    unique_levels = dataFrame[split_column_name].unique()
    combined_dataFrame = None
    for unique_level in unique_levels:
        dataFrame_byLevel = dataFrame.loc[dataFrame[split_column_name] == unique_level, :]
        print "for unique level:", unique_level
        print "data before aggregation:", dataFrame_byLevel.shape
        dataFrame_byLevel = dataFrame_byLevel.drop('app_id', axis = 1).groupby('event_id').aggregate(np.sum)
        ## column name change
        dataFrame_byLevel.columns = update_app_event_column_name(dataFrame_byLevel.columns, unique_level)
        print "data after aggregation:", dataFrame_byLevel.shape
        if combined_dataFrame is None:
            combined_dataFrame = dataFrame_byLevel
        else:
            combined_dataFrame = pd.merge(combined_dataFrame, dataFrame_byLevel, how='outer', left_index=True, right_index=True)    
    return combined_dataFrame

#print "app_event before joining:", app_event.shape
#app_event_labeled = pd.merge(app_event, sparse_app_labels, on='app_id', how='left')
#print "app_event after joining:", app_event_labeled.shape
#agg_app_event_labeled = split_aggregate_combine('is_active', 'event_id', app_event_labeled)

unique_event_ids = app_event['event_id'].unique()
batch_num = 50
agg_app_event_labeled = None

for i in range(batch_num):
    i = 1. * i
    start_index = int(i / batch_num * len(unique_event_ids))
    if i != (batch_num - 1):
        end_index = int((i + 1) / batch_num * len(unique_event_ids))
    else:
        end_index = len(unique_event_ids)

    tmp_app_event = app_event.loc[app_event.event_id.isin(unique_event_ids[start_index:end_index])]
    tmp_app_event_labeled = pd.merge(tmp_app_event, sparse_app_labels, on='app_id', how='left')
    tmp_agg_app_event_labeled = split_aggregate_combine('is_active', 'event_id', tmp_app_event_labeled)
    if agg_app_event_labeled is None:
        agg_app_event_labeled = tmp_agg_app_event_labeled
    else:
        print 'for round:', i, 'the current agg_app_event_labeled shape:', agg_app_event_labeled.shape
        agg_app_event_labeled = pd.concat([agg_app_event_labeled, tmp_agg_app_event_labeled])

print 'agg_app_event_labeled is finished with shape:', agg_app_event_labeled.shape

agg_app_event_labeled_file = 'agg_app_event_labeled_data.csv'
agg_app_event_labeled.to_csv(data_path + agg_app_event_labeled_file)
print 'agg_app_event_labeled is saved into file'


'''
process data on the device_id for events table
'''
events = pd.read_csv(data_path + event_file, index_col='event_id')
events_labeled = pd.merge(events, agg_app_event_labeled, how='left', left_index=True, right_index=True)

print 'the events shape:', events_labeled.shape
print '#unique device_id:', len(events_labeled.device_id.unique())

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


