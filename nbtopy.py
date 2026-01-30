# %%
import pandas as pd
import numpy as np
import streamlit as st
#from itables import show
import hypoxialab_functions as hlab
# from trailing_zeroes import fix_trailing_zeroes, fix_trailing_zeroes_nearest_neighbor, threesamples
from redcap import Project
import io
import os
import psutil

memhist = {}

process = psutil.Process(os.getpid())  # Get the current process (your Streamlit script)
def get_memory_usage():
    """Returns memory usage of the current process in MB"""
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2  # Convert bytes to MB

def print_memory_usage(description):
    """Prints the memory usage of the current process in MB"""
    memhist[description] = get_memory_usage()

# %%
print_memory_usage("Start of script")

SESSION_FORMS = ["hypoxia_lab_session", "sponsor", "study_pi", "device_settings"]  
session = hlab.st_load_project('REDCAP_SESSION', forms=SESSION_FORMS) # to avoid key errors from pulling Labview Samples and Qc Status from redcap session
session = session.reset_index()
manual = hlab.st_load_project('REDCAP_MANUAL') 
participant = hlab.st_load_project('REDCAP_PARTICIPANT')
konica = hlab.st_load_project('REDCAP_KONICA')
devices = hlab.st_load_project('REDCAP_DEVICES')
abg = hlab.st_load_project('REDCAP_ABG').reset_index()
manual = hlab.reshape_manual(manual)
# get the cleaned labview_samples
def get_labview_samples():
    api_url = 'https://redcap.ucsf.edu/api/'
    try:
        api_k = st.secrets['api_k']
    except:
        api_k = os.environ['REDCAP_FILE_RESPOSITORY']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='8', field='file')[0])
    labview_samples = pd.read_csv(f)
    return labview_samples

labview_samples = get_labview_samples()
# only keep the samples with algo_status is True
labview_samples = labview_samples[labview_samples['algo_status']]

print_memory_usage("After loading data")

# start joining the data
manual = manual.rename(columns={'sample_num':'sample', 'session_num':'session'})
manual = manual[['patient_id','device','date','sample', 'session', 'saturation', 'probe_location']]
# the type of sample in manual was object, convert to float64
manual['sample'] = manual['sample'].astype('float64')

# split the device column into a device column and a probe column
manual['probe'] = manual['device'].apply(lambda x: int(str(x).split('.')[1]) if isinstance(x, (float, str)) and '.' in str(x) else None)
manual['device'] = manual['device'].apply(lambda x: int(str(x).split('.')[0]) if isinstance(x, (float, str)) and '.' in str(x) else None)

# merge the session and manual dataframes, so one row is one session with one device. 10 devices in one session = 10 rows
joined = session.merge(manual, left_on=['record_id','patient_id'], right_on=['session','patient_id'], how='right')

# calculate the ITA
konica['ita'] = konica.apply(hlab.ita, axis=1)
# take the median of each unique session
konica_unique_median = konica.groupby(['session','group']).median(numeric_only=True).reset_index()
# keep only Forehead site
konica_unique_median_site = konica_unique_median[konica_unique_median['group'].isin(['Forehead (G)', 'Forehead (E)'])]
# merge the konica data with the joined data
joined = joined.merge(konica_unique_median_site[['session', 'group', 'ita']], on=['session'], how='left')
print_memory_usage("After merging data")

# add participant metadata
participant.reset_index(inplace=True)
participant['subject_id'] = participant['record_id']
participant.drop(columns = ['record_id'], inplace=True)
joined = joined.merge(participant.reset_index(), left_on='patient_id', right_on='subject_id', how='left')

# calculate age at session
joined['session_date'] = pd.to_datetime(joined['session_date'])
joined['dob'] = pd.to_datetime(joined['dob'])
# age at session in years
joined['age_at_session'] = joined['session_date'].dt.year-joined['dob'].dt.year

# # %%
# haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk = hlab.pt_counts(konica,joined)

# #print lenghts of each list in a loop 
# #list descriptions
# desc = ['subjects with konica data', 'subjects who have monk forehead data', 'subjects who have monk forehead data and konica data', 'subjects who have monk forehead data but no konica data', 'subjects who have konica data but no monk forehead data']
# for i,j in zip(desc,[haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk]):
#     print(i,len(j))

# # %% [markdown]
# # Fix the sample number errors in the abg table 

# # %%
# # Delete rows where sample = 0
# abg = abg[abg['sample'] != 0]
# # some so2 values are string for some reason, convert to numeric
# abg['so2'] = pd.to_numeric(abg['so2'], errors='coerce')

# # clean trailing zeroes
# abg, sessions_with_multiple_dates = fix_trailing_zeroes(abg, 8)
# fix_trailing_zeroes_nearest_neighbor(abg, sessions_with_multiple_dates)
# # resolve samples with more than two blood gases per sample
# threesamples(abg)
# print_memory_usage("After cleaning trailing zeroes")
# # if there are 2 samples with the same sample number, average the sao2 and keep only the first sample; 
# # if there are 3 samples with the same sample number, average the sao2 that are not different from more than 0.5
# def average_abl (row):
#     temp = abg[(abg['session'] == row['session']) & (abg['patient_id'] == row['patient_id']) 
#                           & (abg['sample'] == row['sample'])] 
#     if temp.shape[0] == 1:
#         return row['so2']
#     elif temp.shape[0] == 2:
#         return temp['so2'].mean()
#     elif temp.shape[0] == 3:
#         temp = temp.sort_values(by=['time_stamp'])
#         if abs(temp['so2'].iloc[0] - temp['so2'].iloc[1]) <= 0.5:
#             return temp['so2'].iloc[:2].mean()
#         elif abs(temp['so2'].iloc[1] - temp['so2'].iloc[2]) <= 0.5:
#             return temp['so2'].iloc[1:3].mean()
#         elif abs(temp['so2'].iloc[0] - temp['so2'].iloc[2]) <= 0.5:
#             return temp['so2'].iloc[[0,2]].mean()
#     else:
#         return np.nan
    
# abg['so2'] = abg.apply(average_abl, axis=1)

# # drop duplicated rows (keep the first row) for rows with the same session, patient_id, sample
# abg_updated = abg.drop_duplicates(subset=['session', 'patient_id', 'sample'], keep='first')

# %%
# # Merge the joined table with the abg_updated table
# print_memory_usage("Before merging abg and joined")
# abg_updated.loc[:, 'date_calc'] = abg_updated['date_calc'].astype('datetime64[ns]') # convert to datetime so they can merge

print_memory_usage("Before merging labview_samples and joined")
joined = joined.rename(columns ={'date_x':'session_date', 'fitzpatrick_x':'fitzpatrick'})
joined=joined[['patient_id','session_date','session','device','assigned_sex','dob','monk_forehead','monk_dorsal','ita','fitzpatrick', 'sample', 'saturation', 'probe_location']]
joined_updated = pd.merge(joined, labview_samples, on = ['session', 'sample'], how='inner')
print_memory_usage("After merging labview_samples and joined")

# remove encounters with fewer than 16 data points (per device)
# count the non-NaN 'so2' and 'saturation' values for each device-session pair
sample_count_by_device_session = joined_updated.groupby(['device', 'session'])[['so2', 'saturation']].apply(lambda x: x.notna().all(axis=1).sum())
device_session_to_keep = sample_count_by_device_session[sample_count_by_device_session >= 16].index
joined_updated = joined_updated[joined_updated.set_index(['device', 'session']).index.isin(device_session_to_keep)].reset_index()

# exclude sessions in before October 2023
# joined_updated = joined_updated[joined_updated['session_date'] >= '2023-10-01']
# exclude sessions with session_date in June 2023 and September 2023
# joined_updated = joined_updated[~(joined_updated['session_date'].astype(str).str.contains('2023-06')) & ~(joined_updated['session_date'].astype(str).str.contains('2023-09'))]

# %%
haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk = hlab.pt_counts(joined_updated)

#print lenghts of each list in a loop 
#list descriptions
desc = ['subjects with konica data', 'subjects who have monk forehead data', 'subjects who have monk forehead data and konica data', 'subjects who have monk forehead data but no konica data', 'subjects who have konica data but no monk forehead data']
for i,j in zip(desc,[haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk]):
    print(i,len(j))

# %% [markdown]
# Dashboard

#count number of unique patients per device
db = joined_updated[['device','patient_id', 'assigned_sex']].groupby(by=['device','patient_id']).count().reset_index().groupby(by='device').count().reset_index()
db = db.rename(columns={'patient_id':'Unique Subjects'})

#count number of unique sessions per device
tdf = joined_updated[['device','session']].groupby(by=['device','session']).count().reset_index().groupby(by='device').count().reset_index()
tdf = tdf.rename(columns={'session':'Unique Sessions'})
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# # count the number of fitzpatrick in 5 and 6
tdf = joined_updated[joined_updated['fitzpatrick'].isin([5,6])].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
db = db.merge(tdf, left_on='device', right_on='device', how='outer')
db.rename(columns={'patient_id':'Fitzpatrick_5_6'}, inplace=True)

#count assgined_sex
for i in ['Female','Male']:
    tdf = joined_updated[joined_updated['assigned_sex'] == i].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    db.rename(columns={'patient_id':i}, inplace=True)

########## count monk categories
# %%
#define monk skin tone categories
mstlight = ['A','B','C']
mstmedium = ['D','E','F','G']
mstdark = ['H','I','J']
#count those with monk forehead that is light, medium, or dark
for i in [mstlight, mstmedium, mstdark]:
    # select only those with monk forehead that is light, medium, or dark
    # groupby device and patient id to get each unique device-patient pair
    # then groupby device to get the count of unique patients per device
    tdf = joined_updated[joined_updated['monk_forehead'].isin(i)].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
    # merge the new monk forehead count data with the dashboard frame
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    db.rename(columns={'patient_id':'monk_forehead_'+i[0]}, inplace=True)

mst_1_2 = ['A', 'B']
mst_3_4 = ['C', 'D']
mst_5_6 = ['E', 'F']
mst_7_8 = ['G', 'H']
mst_9_10 = ['I', 'J']

for i in [mst_1_2, mst_3_4, mst_5_6, mst_7_8, mst_9_10]:
    tdf = joined_updated[joined_updated['monk_forehead'].isin(i)].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    db.rename(columns={'patient_id':'monk_forehead_'+i[0]+i[1]}, inplace=True)

# check if >= 1 in each of the 10 MST categories
# check the number of unique monk_forehead per device
tdf = joined_updated[joined_updated['monk_forehead'].notnull()]
tdf = joined_updated.groupby(by=['device']).nunique()['monk_forehead'].reset_index()
tdf.rename(columns={'monk_forehead':'unique_monk_forehead'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# check the number of unique monk_dorsal per device
tdf = joined_updated[joined_updated['monk_dorsal'].notnull()]
tdf = joined_updated.groupby(by=['device']).nunique()['monk_dorsal'].reset_index()
tdf.rename(columns={'monk_dorsal':'unique_monk_dorsal'}, inplace=True)
db = db.merge(tdf, on = 'device', how='outer')

# display the unique monk_forehead values for each device
tdf = joined_updated[joined_updated['monk_forehead'].notnull()]
tdf = tdf.groupby(by=['device'])['monk_forehead'].unique().reset_index()
# sort the monk_forehead values per device
tdf['monk_forehead'] = tdf['monk_forehead'].apply(lambda x: sorted(x))
tdf.rename(columns={'monk_forehead':'Unique Monk Forehead Values'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# display the unique monk_dorsal values for each device
tdf = joined_updated[joined_updated['monk_dorsal'].notnull()]
tdf = tdf.groupby(by=['device'])['monk_dorsal'].unique().reset_index()
# sort the monk_dorsal values per device
tdf['monk_dorsal'] = tdf['monk_dorsal'].apply(lambda x: sorted(x))
tdf.rename(columns={'monk_dorsal':'Unique Monk Dorsal Values'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

########## count ITA categories
# ITA criteria: >=50, <=-45, >30, between 30 to -30, <-=-20, <=-50
itacriteria = [(joined_updated['ita']>=50) & (joined_updated['monk_forehead'].isin(['A','B','C','D'])), (joined_updated['ita']<=-45) & (joined_updated['monk_forehead'].isin(['H','I','J'])), (joined_updated['ita']>30) & (joined_updated['monk_forehead'].isin(['A','B','C'])), (joined_updated['ita']<=30) & (joined_updated['ita']>=-30) & (joined_updated['monk_forehead'].isin(['D', 'E', 'F', 'G'])), (joined_updated['ita']<-30) & (joined_updated['monk_forehead'].isin(['H','I','J'])), (joined_updated['ita']<-50) & (joined_updated['monk_forehead'].isin(['H','I','J']))]
criterianames = ['ita>=50&MonkABCD','ita<=-45&MonkHIJ','ita>30&MonkABC','ita30to-30&MonkDEFG','ita<-30&MonkHIJ', 'ita<-50&MonkHIJ']

for i,j in zip(itacriteria,criterianames):
    # lily is interested in the SID for subject with ITA < -50 and Monk HIJ
    if j == 'ita<-50&MonkHIJ':
        tdf = joined_updated[i].groupby(by='device')['patient_id'].unique().reset_index()
        tdf['patient_id'] = tdf['patient_id'].apply(lambda x: sorted(x))
        tdf = tdf[['device', 'patient_id']].set_index('device').rename(columns={'patient_id':'ITA < -50 & Monk HIJ SID'})
        db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    # temp dataframe for each device, counting only the patients who meet the criteria
    tdf = joined_updated[i].groupby(by=['device','patient_id']).count().reset_index().groupby(by='device').count().reset_index()
    # select only device and patient id columns, rename patient id to the criteria name
    tdf = tdf[['device','patient_id']].set_index('device').rename(columns={'patient_id':j})
    # merge with dashboard frame
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
########## add age at session

# db = db.merge(stats('age_at_session',joined_updated), left_on='device', right_index=True, how='outer')
# db = db.merge(stats('bmi',joined_updated), left_on='device', right_index=True, how='outer')

########## add device names and priority
# merge with devices, keeping all devices
db = devices[['manufacturer','model','priority']].merge(db, left_index=True, right_on='device', how='outer').reset_index().drop(columns=['index'])
db = db.rename(columns={'manufacturer':'Manufacturer', 'model':'Model'})

########## add abg caculations
# # check average number of samples per session
# abg_updated_max = abg_updated.groupby(['patient_id','session']).max('sample').reset_index()[['patient_id','session','sample']]
# abg_updated_max.rename(columns={'sample':'max_sample'}, inplace=True)
# # merge the joined_updated table with the abg_updated_max table
# joined_updated = joined_updated.merge(abg_updated_max, left_on=['patient_id','session'], right_on=['patient_id','session'], how='left')
# # create a temp dataframe that has the average of 'sample_max' in each device
# tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['max_sample'].reset_index()
# tdf.rename(columns={'max_sample':'avg_sample'}, inplace=True)
# tdf['avg_sample'] = tdf['avg_sample'].round(2)
# db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# # check range of number of samples per session, if the session satistifes the criteria, label as 1
# joined_updated['sample_range'] = joined_updated['max_sample'].apply(lambda x: 1 if (x >= 16) & (x <= 30) else 0)
# # count the number of unique patient that have sample_range = 1 per device
# tdf = joined_updated[joined_updated['sample_range'] == 1].groupby(by=['device']).nunique()['patient_id'].reset_index()
# tdf.rename(columns={'patient_id':'sample_range'}, inplace=True)
# db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# # check if >= 90% of the sessions in the same device provide so2 data < 85
# # create a new column called 'so2<85' that is 1 if so2 < 85, 0 if so2 >= 85 in abg_updated table
# abg_updated_copy = abg_updated.copy()
# abg_updated_copy['so2<85'] = abg_updated_copy['so2'].apply(lambda x: 1 if x < 85 else 0)
# # group the abg_updated table by session, and calculate the mean of 'so2<85' in each session
# abg_updated_copy['session'] = abg_updated_copy['session'].astype(str)
# abg_updated_copy = abg_updated_copy.groupby(by=['session']).mean(numeric_only=True)['so2<85'].reset_index()
# # change the so2<85 column to 1 if the column value is greater than 0, 0 otherwise
# abg_updated_copy['so2<85'] = abg_updated_copy['so2<85'].apply(lambda x: 1 if x > 0 else 0)
# # merge the abg_updated table with the joined_updated table
# abg_updated_copy = abg_updated_copy.dropna()
# abg_updated_copy['session'] = abg_updated_copy['session'].astype(float)
# joined_updated = joined_updated.merge(abg_updated_copy, on=['session'], how='left')
# # group the so2<85 column by device, and calculate the mean of so2<85 in each device
# tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['so2<85'].reset_index()
# tdf['so2<85'] = tdf['so2<85'].apply(lambda x: x*100)
# db = db.merge(tdf, on ='device', how='outer')

# # check if >=70% participants/sessions provide data points in the 70%(-3%) - 80% decade (sao2)
# abg_updated_copy = abg_updated.copy()
# abg_updated_copy['sao2_70-80'] = abg_updated_copy['so2'].apply(lambda x: 1 if (x >= 67) & (x <= 80) else 0)
# abg_updated_copy['session'] = abg_updated_copy['session'].astype(str)
# abg_updated_copy = abg_updated_copy.groupby(by=['session']).mean(numeric_only=True)['sao2_70-80'].reset_index()
# abg_updated_copy['sao2_70-80'] = abg_updated_copy['sao2_70-80'].apply(lambda x: 1 if x > 0 else 0)
# abg_updated_copy = abg_updated_copy.dropna()
# abg_updated_copy['session'] = abg_updated_copy['session'].astype(float)
# joined_updated = joined_updated.merge(abg_updated_copy, on=['session'], how='left')
# tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['sao2_70-80'].reset_index()
# tdf['sao2_70-80'] = tdf['sao2_70-80'].apply(lambda x: x*100)
# db = db.merge(tdf, left_on='device', right_on='device', how='outer')

######### add sample-related caculations
# check average number of samples per session
labview_samples_max = labview_samples.groupby(['session']).max('sample').reset_index()[['session','sample']]
labview_samples_max.rename(columns={'sample':'max_sample'}, inplace=True)
# merge the joined_updated table with the labview_samples_max table
joined_updated = joined_updated.merge(labview_samples_max, left_on=['session'], right_on=['session'], how='left')
# create a temp dataframe that has the average of 'sample_max' in each device
tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['max_sample'].reset_index()
tdf.rename(columns={'max_sample':'avg_sample'}, inplace=True)
tdf['avg_sample'] = tdf['avg_sample'].round(2)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# check range of number of samples per session, if the session satistifes the criteria, label as 1
joined_updated['sample_range'] = joined_updated['max_sample'].apply(lambda x: 1 if (x >= 16) else 0)
# count the number of unique patient that have sample_range = 1 per device
tdf = joined_updated[joined_updated['sample_range'] == 1].groupby(by=['device']).nunique()['patient_id'].reset_index()
tdf.rename(columns={'patient_id':'sample_range'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# check if >= 90% of the sessions in the same device provide so2 data < 85
# create a new column called 'so2<85' that is 1 if so2 < 85, 0 if so2 >= 85 in labview_samples table
labview_samples_copy = labview_samples.copy()
labview_samples_copy['so2<85'] = labview_samples_copy['so2'].apply(lambda x: 1 if x < 85 else 0)
# group the labview_samples table by session, and calculate the mean of 'so2<85' in each session
labview_samples_copy['session'] = labview_samples_copy['session'].astype(str)
labview_samples_copy = labview_samples_copy.groupby(by=['session']).mean(numeric_only=True)['so2<85'].reset_index()
# change the so2<85 column to 1 if the column value is greater than 0, 0 otherwise
labview_samples_copy['so2<85'] = labview_samples_copy['so2<85'].apply(lambda x: 1 if x > 0 else 0)
# merge the labview_samples table with the joined_updated table
labview_samples_copy = labview_samples_copy.dropna()
labview_samples_copy['session'] = labview_samples_copy['session'].astype(float)
joined_updated = joined_updated.merge(labview_samples_copy, on=['session'], how='left')
# group the so2<85 column by device, and calculate the mean of so2<85 in each device
tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['so2<85'].reset_index()
tdf['so2<85'] = tdf['so2<85'].apply(lambda x: x*100)
db = db.merge(tdf, on ='device', how='outer')

# check if >=70% participants/sessions provide data points in the 70%(-3%) - 80% decade (sao2)
labview_samples_copy = labview_samples.copy()
labview_samples_copy['sao2_70-80'] = labview_samples_copy['so2'].apply(lambda x: 1 if (x >= 67) & (x <= 80) else 0)
labview_samples_copy['session'] = labview_samples_copy['session'].astype(str)
labview_samples_copy = labview_samples_copy.groupby(by=['session']).mean(numeric_only=True)['sao2_70-80'].reset_index()
labview_samples_copy['sao2_70-80'] = labview_samples_copy['sao2_70-80'].apply(lambda x: 1 if x > 0 else 0)
labview_samples_copy = labview_samples_copy.dropna()
labview_samples_copy['session'] = labview_samples_copy['session'].astype(float)
joined_updated = joined_updated.merge(labview_samples_copy, on=['session'], how='left')
tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['sao2_70-80'].reset_index()
tdf['sao2_70-80'] = tdf['sao2_70-80'].apply(lambda x: x*100)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

### add PM1000N PI to dashboard
# organize the Device 59 (Nellcor PM1000N) and Device 60 (Masimo Rad-97) PI to the same column
joined_updated['Masimo 97/PI'] = joined_updated.apply(lambda row: row['Masimo 97/PI'] if pd.notnull(row['Masimo 97/PI']) else (row['Masimo HB/PI'] if pd.notnull(row['Masimo HB/PI']) else row['Rad97-60/PI']), axis=1)
joined_updated['Nellcor/PI'] = joined_updated.apply(lambda row: row['Nellcor/PI'] if pd.notnull(row['Nellcor/PI']) else row['Nellcor PM1000N-1/PI'], axis=1)
joined_updated['Nellcor/PI'] = joined_updated['Nellcor/PI'] / 10
# report median and Q1, Q3 of Nellcor PI
tdf = joined_updated.groupby('device')['Nellcor/PI'].agg(lambda x: f"{x.median():.2f} ({x.quantile(0.25):.2f}, {x.quantile(0.75):.2f})").reset_index()
tdf.rename(columns={'Nellcor/PI': 'PI Median (Q1, Q3)'}, inplace=True)
# merge the result back into the original dataframe
db = db.merge(tdf, on='device', how='outer')

# # groupby device and session, and count of number of session per device that have median Nellcor/PI <1
# # First, group by device and session, and calculate the median Nellcor/PI for each session
# tdf = joined_updated.groupby(['device', 'session'])['Nellcor/PI'].median().reset_index()
# # Create a new column to flag sessions where the median Nellcor/PI is less than 1
# tdf['PI_less_than_1'] = tdf['Nellcor/PI'] < 1
# # Now group by device and calculate the number of sessions with PI < 1 and the total number of sessions
# tdf = tdf.groupby('device').agg(
#     num_sessions_with_PI_less_than_1=('PI_less_than_1', 'sum'),  # Count sessions with PI < 1
#     total_sessions=('session', 'size')  # Total number of sessions per device
# ).reset_index()
# # Add a percentage column
# tdf['percentage'] = (tdf['num_sessions_with_PI_less_than_1'] / tdf['total_sessions']) * 100
# # Create the final column as "number of sessions (percentage)"
# tdf['Count of Session with PI Median<1 (%)'] = tdf.apply(
#     lambda row: f"{row['num_sessions_with_PI_less_than_1']} ({row['percentage']:.2f}%)", axis=1
# )
# # Merge this information back into the `db` dataframe
# db = db.merge(tdf[['device', 'Count of Session with PI Median<1 (%)']], on='device', how='left')
# # Now `db` has a new column `sessions_summary` that includes the count and percentage of sessions with PI < 1
# print(db[['device', 'Count of Session with PI Median<1 (%)']].head())

# percentage of PI<1, PI1-2, and PI>2
def categorize_pi(pi_value):
    if pd.isna(pi_value):
        return 'NaN'
    elif pi_value < 1:
        return 'PI < 1'
    elif 1 <= pi_value <= 2:
        return '1 ≤ PI ≤ 2'
    else:
        return 'PI > 2'

joined_updated['nellcor_pi_category'] = joined_updated['Nellcor/PI'].apply(categorize_pi)
# group by device and calculate the percentage of each category
pi_percentage = joined_updated[joined_updated['nellcor_pi_category'] != 'NaN'].groupby('device')['nellcor_pi_category'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
pi_percentage = pi_percentage.rename(columns={
    'PI < 1': '% PI <1',
    '1 ≤ PI ≤ 2': '% 1 ≤ PI ≤ 2',
    'PI > 2': '% PI >2'
})
# Apply formatting column-wise using map
for col in pi_percentage.columns:
    pi_percentage[col] = pi_percentage[col].map(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))
# pi_percentage = pi_percentage.applymap(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))
# Merge these percentages back into the original 'db' DataFrame on the 'device' column
db = db.merge(pi_percentage, on='device', how='left')

# finger location count
group_columns_mapping = {
    'thumb': [1.0, 6.0],
    'index_finger': [2.0, 7.0],
    'middle_finger': [3.0, 8.0],
    'ring_finger': [4.0, 9.0],
    'pinky_finger': [5.0, 10.0]
}

def calculate_percentage(group):
    # Total number of rows per device
    total_count = group['probe_location'].count()
    
    # Create a dictionary to hold percentages for each finger group
    percentage_dict = {}
    
    # For each group of probe locations (thumb, index finger, etc.)
    for finger, values in group_columns_mapping.items():
        count = group[group['probe_location'].isin(values)].shape[0]
        percentage_dict[f'% in {finger.capitalize()}'] = int(round((count / total_count) * 100))
    
    return pd.Series(percentage_dict)

# Group by device and calculate the percentages
percentages_df = joined_updated.groupby('device').apply(calculate_percentage).reset_index()
percentages_df.rename(columns=lambda x: x.replace('_', ' ').title() if '%' in x else x, inplace=True)
# print(percentages_df.iloc[:, 1:].dtypes)
# Apply formatting column-wise using DataFrame.map
for col in percentages_df.columns[1:]:
    percentages_df[col] = percentages_df[col].map(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))
db = pd.merge(db, percentages_df, on='device', how='left')

# %%
# Check if each decade between the 70% - 100% saturations contains 33% of the data points (sao2)
# group the joined_updated table by device, create three new columns called 'so2_70-80', 'so2_80-90', 'so2_90-100' that is the count of so2 in each decade
tdf = joined_updated[(joined_updated['so2'] >= 67) & (joined_updated['so2'] < 100)].groupby(by=['device']).count()['so2'].reset_index()
tdf.rename(columns={'so2':'total'}, inplace=True)
t = joined_updated[(joined_updated['so2'] >= 67) & (joined_updated['so2'] < 80)].groupby(by=['device']).count()['so2'].reset_index().rename(columns={'so2':'so2_70-80'})
tdf = pd.merge(tdf, t, on='device', how='left')
t = joined_updated[(joined_updated['so2'] >= 80) & (joined_updated['so2'] < 90)].groupby(by=['device']).count()['so2'].reset_index().rename(columns={'so2':'so2_80-90'})
tdf = pd.merge(tdf, t, on='device', how='left')
t = joined_updated[(joined_updated['so2'] >= 90) & (joined_updated['so2'] < 100)].groupby(by=['device']).count()['so2'].reset_index().rename(columns={'so2':'so2_90-100'})
tdf = pd.merge(tdf, t, on='device', how='left')
# calculate the percentage of each decade
tdf['so2_70-80'] = round(tdf['so2_70-80']/tdf['total'], 2) * 100
tdf['so2_80-90'] = round(tdf['so2_80-90']/tdf['total'], 2) * 100
tdf['so2_90-100'] = round(tdf['so2_90-100']/tdf['total'], 2) * 100
tdf = tdf.drop(columns=['total'])
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# group by device and find the min and max of so2 as two columns in the tdf dataframe
tdf = joined_updated.groupby(by=['device']).agg({'so2':['min','max']}).reset_index()
tdf.columns = ['device','min_sao2','max_sao2']
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# group by device and count the number of sessions with >=25% of so2 data points in the 70%-80%, 80%-90%, and 90% above decade respectively
tdf = joined_updated[(joined_updated['so2'] >= 67) & (joined_updated['so2'] < 100)].groupby(by=['device','session']).count()['so2'].reset_index()
tdf.rename(columns={'so2':'total'}, inplace=True)
t = joined_updated[(joined_updated['so2'] >= 67) & (joined_updated['so2'] < 80)].groupby(by=['device','session']).count()['so2'].reset_index().rename(columns={'so2':'so2_70-80'})
tdf = pd.merge(tdf, t, on=['device','session'], how='left')
t = joined_updated[(joined_updated['so2'] >= 80) & (joined_updated['so2'] < 90)].groupby(by=['device','session']).count()['so2'].reset_index().rename(columns={'so2':'so2_80-90'})
tdf = pd.merge(tdf, t, on=['device','session'], how='left')
t = joined_updated[(joined_updated['so2'] >= 90) & (joined_updated['so2'] < 100)].groupby(by=['device','session']).count()['so2'].reset_index().rename(columns={'so2':'so2_90-100'})
tdf = pd.merge(tdf, t, on=['device','session'], how='left')
# calculate the percentage of each decade
tdf['so2_70-80'] = round(tdf['so2_70-80']/tdf['total'], 2) * 100
tdf['so2_80-90'] = round(tdf['so2_80-90']/tdf['total'], 2) * 100
tdf['so2_90-100'] = round(tdf['so2_90-100']/tdf['total'], 2) * 100
tdf = tdf.drop(columns=['total'])
# for each row in tdf, drop the row if any of the three decades is less than 25
tdf = tdf[(tdf['so2_70-80'] >= 25) & (tdf['so2_80-90'] >= 25) & (tdf['so2_90-100'] >= 25)]
# group by device and count the number of sessions in tdf
tdf = tdf.groupby(by=['device']).count()['session'].reset_index()
tdf.rename(columns={'session':'session_count'}, inplace=True)
# merge db with tdf, if there is no session with >=25% of so2 in the 3 decades, fill the value with 0
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# %%
# Replace NaN values in some specific columns with empty lists - to avoid raising warnings
db['Unique Monk Forehead Values'] = db['Unique Monk Forehead Values'].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else (x if isinstance(x, list) else [x]))
db['Unique Monk Dorsal Values'] = db['Unique Monk Dorsal Values'].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else (x if isinstance(x, list) else [x]))
db['ITA < -50 & Monk HIJ SID'] = db['ITA < -50 & Monk HIJ SID'].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else (x if isinstance(x, list) else [x]))
# fill zeroes
db.fillna(0, inplace=True)

print(f"Device 81 has {db[db['device']==81]['Unique Subjects'].iloc[0]} unique subjects.")

db_new_v1 = db[['Manufacturer', 'Model', 'priority', 'device', 'Unique Subjects', 'Female', 'Male', 'monk_forehead_A', 'monk_forehead_D', 'monk_forehead_H', 'unique_monk_forehead', 'Unique Monk Forehead Values', 'unique_monk_dorsal', 'Unique Monk Dorsal Values', 'ita>=50&MonkABCD', 'ita<=-45&MonkHIJ', 'ita>30&MonkABC', 'ita30to-30&MonkDEFG', 'ita<-30&MonkHIJ', 'ita<-50&MonkHIJ', 'ITA < -50 & Monk HIJ SID', 'avg_sample', 'sample_range', 'min_sao2', 'max_sao2', 'so2<85', 'sao2_70-80', 'so2_70-80', 'so2_80-90', 'so2_90-100', 'session_count']]
#create a dictionary of column names and their descriptions
column_dict_db_new_v1 = {'device':'Device',
                'Unique Subjects':'Unique Subjects',
                'Female': 'Female',
                'Male': 'Male',
                'monk_forehead_A':'Monk ABC',
                'monk_forehead_D':'Monk DEFG',
                'monk_forehead_H':'Monk HIJ',
                'unique_monk_forehead':'Unique Monk Forehead',
                'unique_monk_dorsal':'Unique Monk Dorsal',
                'ita>=50&MonkABCD':'ITA >= 50 & Monk ABCD',
                'ita<=-45&MonkHIJ':'ITA <= -45 & Monk HIJ',
                'ita>30&MonkABC':'ITA > 30 & Monk ABC',
                'ita30to-30&MonkDEFG':'-30 <= ITA <= 30 & Monk DEFG',
                'ita<-30&MonkHIJ':'ITA < -30 & Monk HIJ',
                'ita<-50&MonkHIJ': 'ITA < -50 & Monk HIJ',
                'priority':'Test Priority',
                'avg_sample':'Avg Samples per Session',
                'sample_range':'Unique Subjects with >=16 Samples',
                'so2<85':'%\n of Sessions Provides SaO2 < 85',
                "min_sao2":'Min SaO2',
                "max_sao2":'Max SaO2',
                'sao2_70-80':'%\n of Sessions Provides SaO2 in 70-80',
                'so2_70-80':'%\n of SaO2 in 70-80 (pooled)',
                'so2_80-90':'%\n of SaO2 in 80-90 (pooled)',
                'so2_90-100':'%\n of SaO2 in 90-100 (pooled)',
                'session_count':'# of Sessions with >=25%\n of SaO2 in 70-80, 80-90, 90-100'
                }

db_new_v2 = db[['Manufacturer', 'Model', 'priority', 'device', 'Unique Subjects', 'Female', 'Male', 'monk_forehead_A', 'monk_forehead_D', 'monk_forehead_H', 'monk_forehead_AB', 'monk_forehead_CD', 'monk_forehead_EF', 'monk_forehead_GH', 'monk_forehead_IJ', 'ita<-50&MonkHIJ', 'unique_monk_forehead', 'Unique Monk Forehead Values', 'unique_monk_dorsal', 'Unique Monk Dorsal Values', 'avg_sample', 'sample_range', 'min_sao2', 'max_sao2', 'so2<85', 'sao2_70-80', 'so2_70-80', 'so2_80-90', 'so2_90-100', 'session_count']]
#create a dictionary of column names and their descriptions
column_dict_db_new_v2 = {'device':'Device',
                'Unique Subjects':'Unique Subjects',
                'Female': 'Female',
                'Male': 'Male',
                'monk_forehead_A':'Monk ABC',
                'monk_forehead_D':'Monk DEFG',
                'monk_forehead_H':'Monk HIJ',
                'monk_forehead_AB':'Monk AB',
                'monk_forehead_CD':'Monk CD',
                'monk_forehead_EF':'Monk EF',
                'monk_forehead_GH':'Monk GH',
                'monk_forehead_IJ':'Monk IJ',
                'ita<-50&MonkHIJ': 'ITA < -50 & Monk HIJ',
                'unique_monk_forehead':'Unique Monk Forehead',
                'unique_monk_dorsal':'Unique Monk Dorsal',
                'priority':'Test Priority',
                'avg_sample':'Avg Samples per Session',
                'sample_range':'Unique Subjects with >=16 Samples',
                "min_sao2":'Min SaO2',
                "max_sao2":'Max SaO2',
                'so2<85':'%\n of Sessions Provides SaO2 < 85',
                'sao2_70-80':'%\n of Sessions Provides SaO2 in 70-80',
                'so2_70-80':'%\n of SaO2 in 70-80 (pooled)',
                'so2_80-90':'%\n of SaO2 in 80-90 (pooled)',
                'so2_90-100':'%\n of SaO2 in 90-100 (pooled)',
                'session_count':'# of Sessions with >=25%\n of SaO2 in 70-80, 80-90, 90-100'
                }

db_old = db[['Manufacturer', 'Model', 'priority', 'device', 'Unique Sessions', 'Fitzpatrick_5_6', 'avg_sample', 'sample_range', 'min_sao2', 'max_sao2', 'so2_70-80', 'so2_80-90', 'so2_90-100']]
# fill zeroes
db_old = db_old.fillna(0)
#create a dictionary of column names and their descriptions
column_dict_db_old = {'device':'Device',
                'priority':'Test Priority',
                'Unique Sessions':'Unique Sessions',
                'Fitzpatrick_5_6':'# Sessions with Fitzpatrick V or VI',
                'avg_sample':'Avg Samples per Session',
                'sample_range':'Unique Sessions with >= 20 Samples',
                "min_sao2":'Min SaO2',
                "max_sao2":'Max SaO2',
                'so2_70-80':'%\n of SaO2 in 70-80 (pooled)',
                'so2_80-90':'%\n of SaO2 in 80-90 (pooled)',
                'so2_90-100':'%\n of SaO2 in 90-100 (pooled)',
                }


db_new_v3 = db[['Manufacturer', 'Model', 'priority', 'device', 'Unique Subjects', 'Female', 'Male', 'ita>30&MonkABC', 'ita30to-30&MonkDEFG', 'ita<-30&MonkHIJ', 'ita<-50&MonkHIJ', 'ITA < -50 & Monk HIJ SID', 'avg_sample', 'sample_range', 'min_sao2', 'max_sao2', 'so2<85', 'sao2_70-80', 'so2_70-80', 'so2_80-90', 'so2_90-100', 'PI Median (Q1, Q3)', '% PI <1', '% 1 ≤ PI ≤ 2', '% PI >2', '% In Thumb', '% In Index Finger',
       '% In Middle Finger', '% In Ring Finger', '% In Pinky Finger']]
#create a dictionary of column names and their descriptions
column_dict_db_new_v3 = {'device':'Device',
                        'priority':'Test Priority',
                        'Unique Subjects':'Unique Subjects',
                        'Female': 'Female',
                        'Male': 'Male',
                        'ita>30&MonkABC':'ITA > 30 & Monk ABC',
                        'ita30to-30&MonkDEFG':'-30 <= ITA <= 30 & Monk DEFG',
                        'ita<-30&MonkHIJ':'ITA < -30 & Monk HIJ',
                        'ita<-50&MonkHIJ': 'ITA < -50 & Monk HIJ',
                        'avg_sample':'Avg Samples per Session',
                        'sample_range':'Unique Subjects with >=16 Samples',
                        'so2<85':'%\n of Sessions Provides SaO2 < 85',
                        "min_sao2":'Min SaO2',
                        "max_sao2":'Max SaO2',
                        'sao2_70-80':'%\n of Sessions Provides SaO2 in 70-80',
                        'so2_70-80':'%\n of SaO2 in 70-80 (pooled)',
                        'so2_80-90':'%\n of SaO2 in 80-90 (pooled)',
                        'so2_90-100':'%\n of SaO2 in 90-100 (pooled)'
                        }

print_memory_usage("After dashboard")