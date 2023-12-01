# %%
import pandas as pd
import numpy as np
import os
from redcap import Project
import streamlit as st
#from itables import show
import math

# %% [markdown]
# Functions

# %%
#load all data from redcap
def load_project(key):
    # api_key = config.redcap[key]
    api_key = os.environ.get(key)
    api_url = 'https://redcap.ucsf.edu/api/'
    project = Project(api_url, api_key)
    df = project.export_records(format_type='df')
    return df

def st_load_project(key):
    api_key = st.secrets[key]
    api_url = 'https://redcap.ucsf.edu/api/'
    project = Project(api_url, api_key)
    df = project.export_records(format_type='df')
    return df

#reshape manual entered data into long format
def reshape_manual(df):

    reshaped=pd.DataFrame()

    for index, row in df.iterrows(): # iterate through every patient
        for i in range(1,11): # iterate through each device in every patient
            # create temp df from the row containing only device information
            t2 = row.filter(regex=f'{i}$') 
            t2 = pd.DataFrame(t2)

            #label the sample number from the index
            t2['sample_num'] = t2.index
            t2['sample_num'] = t2['sample_num'].str.extract(r'sat(\d+)') 

            #within each row, label the device
            t2['device'] = row[f'dev{i}'] 

            #within each row, label the location
            t2['probe_location'] = row[f'loc{i}'] 
            
            #etc
            t2['date'] = row['date']
            t2['session_num'] = row['session']
            t2['patient_id'] = row['patientid']

            #drop the columns not relating to saturation, device and location
            t2 = t2.drop([f'dev{i}', f'loc{i}']) 
            
            #label first column as saturation
            t2.columns.values[0] = 'saturation'

            #concatenate
            reshaped = pd.concat([reshaped, t2], axis=0)

    reshaped=reshaped[reshaped['saturation'].notnull()]
    return reshaped

# count patients
def pt_counts(konica,joined):
    # list of patients with konica data
    haskonica = konica['upi'].unique().tolist()
    #patients who have monk dorsal data
    hasmonk = joined[joined['monk_dorsal'].notnull()]['patient_id'].unique().tolist()
    #patients who have monk dorsal data and konica data
    hasboth = list(set(haskonica) & set(hasmonk))
    #patients who have monk dorsal data but no konica data
    hasmonk_notkonica = list(set(hasmonk) - set(haskonica))
    #patients who have konica data but no monk dorsal data
    haskonica_notmonk = list(set(haskonica) - set(hasmonk))
    return haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk

# calculate numeric stats
def stats(var, joined):
    stats = []
    stats.append(joined.groupby(by=['device','patient_id']).mean(numeric_only=True)[var].reset_index().groupby(by='device').mean()[var])
    stats.append(joined.groupby(by=['device','patient_id']).min(numeric_only=True)[var].reset_index().groupby(by='device').min()[var])
    stats.append(joined.groupby(by=['device','patient_id']).max(numeric_only=True)[var].reset_index().groupby(by='device').max()[var])

    stats = pd.concat(stats, axis=1)

    stats.columns = ['mean_'+var,'min_'+var,'max_'+var]
    stats[var+'_range'] = stats['max_'+var]-stats['min_'+var]
    return stats

def ita(row):
    return (np.arctan((row['lab_l']-50)/row['lab_b'])) * (180/math.pi)

# style database
def highlight_value_greater(s, cols_to_sum,threshold):
    sums = db[cols_to_sum].sum(axis=1)
    mask = (s >= sums*threshold) & (sums > 0)
    return ['background-color: green' if v else '' for v in mask]


# %%

session = st_load_project('REDCAP_SESSION')
session = session.reset_index()
manual = st_load_project('REDCAP_MANUAL') 
participant = st_load_project('REDCAP_PARTICIPANT')
konica = st_load_project('REDCAP_KONICA')
devices = st_load_project('REDCAP_DEVICES')
abg = st_load_project('REDCAP_ABG').reset_index()
manual = reshape_manual(manual)

# keep only device and date columns from manual
manual = manual[['patient_id','device','date']]

# merge the session and manual dataframes, so one row is one session with one device. 10 devices in one session = 10 rows
joined = session.merge(manual, left_on=['patient_id','session_date'], right_on=['patient_id','date'], how='left')

# take the median of each unique session
konica_unique_median = konica.groupby(['date','upi','group']).median(numeric_only=True).reset_index()
# calculate the ITA
konica_unique_median['ita'] = konica_unique_median.apply(ita, axis=1)
# keep only dorsal site
konica_unique_median_site = konica_unique_median[konica_unique_median['group'].str.contains(r'\WB\W')]

# merge the konica data with the session data
joined = joined.merge(konica_unique_median_site, left_on=['patient_id','session_date'], right_on=['upi','date'], how='left')

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

# %%
haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk = pt_counts(konica,joined)

#print lenghts of each list in a loop 
#list descriptions
desc = ['subjects with konica data', 'subjects who have monk dorsal data', 'subjects who have monk dorsal data and konica data', 'subjects who have monk dorsal data but no konica data', 'subjects who have konica data but no monk dorsal data']
for i,j in zip(desc,[haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk]):
    print(i,len(j))

# %%
# how many individual patients and their ITA ranges?
t1 = konica_unique_median.groupby(by=['upi']).agg({'ita':['min','max']}).reset_index()
#t1[t1['patient_id']==958].reset_index()
t1[t1['ita']['min']<-45]
# t1

# %% [markdown]
# Fix the sample number errors in the abg table 

# %%
# 1. Delete rows where sample = 0
abg = abg[abg['sample'] != 0]

# check the number of rows in abg - 8271 -> 8268
# abg.shape[0] 

# 2. Edit the trimmed trailing zeros issue
abg['time_stamp'] = pd.to_datetime(abg['time_stamp'])

def update_sample(row):
    if row['sample'] not in [1,2,3]:
        return row['sample']
    temp = abg[(abg['session'] == row['session']) & (abg['patient_id'] == row['patient_id']) 
                          & (abg['sample'] == row['sample'])] 
    # check the sample timestamp is at least 5 minutes after the timestamp of the first sample with that label
    if (row['time_stamp'] - temp['time_stamp'].min()).seconds > 300:
        # if yes, return the 'sample number' + '0'
        return row['sample']*10
    return row['sample']

# make a copy of abg
abg_updated = abg.copy()
abg_updated['sample'] = abg_updated.apply(update_sample, axis=1)

# %%
# Merge the joined table with the abg_updated table
abg_updated['date_calc'] = abg_updated['date_calc'].astype('datetime64[ns]') # convert to datetime so they can merge
joined_updated = pd.merge(joined, abg_updated.rename(columns ={'date_calc':'session_date'}), left_on = ['patient_id', 'session_date', 'session'], right_on = ['patient_id', 'session_date','session'], how='left')

# %%
# print out session 339 in abg and updated abg to view the changes 
# abg[(abg['session'] == 339) & ((abg['sample'] == 1) | (abg['sample'] == 2) | (abg['sample'] == 3) | (abg['sample'] == 10) | (abg['sample'] == 20) | (abg['sample'] == 30)) ].sort_values(['time_stamp'], ascending=True)
# abg_updated[(abg_updated['session'] == 339) & ((abg_updated['sample'] == 1) | (abg_updated['sample'] == 2) | (abg_updated['sample'] == 3) | (abg_updated['sample'] == 10) | (abg_updated['sample'] == 20) | (abg_updated['sample'] == 30)) ].sort_values(['time_stamp'], ascending=True)

# %%
# Check if there are any patient_id that has the same session number -> found (1110, 1100) and (1098,1022)
abg_2 = abg.copy()
#######identify duplicate session/patient_id combinations
dupes = abg_2.groupby(['patient_id','session']).size().reset_index()['session'].value_counts() # get list of unique session/patient_id combinations, and count how many times each session appears
dupes = dupes[dupes > 1].index.tolist() # get the session numbers that appear more than once
dupes = abg_2[abg_2['session'].isin(dupes)][['patient_id','session']] # filter the abg table to only include the session numbers that appear more than once
if len(dupes) > 0:
    st.write(dupes)
else:
    st.write('No duplicate session/patient_id combinations')


####################DANNI-> what does this section do? and do we need it?####################
####################TYLER -> This is just me trying to check if there is any more error with diff patient_id having the same session ID. We can delete this as the issue is fixed. ########################
# set col patient_ids to be all patient_ids for each session
abg_2['patient_id'] = abg_2['patient_id'].astype(str)
abg_2 = abg_2[['session', 'patient_id']].groupby(['session'])['patient_id'].transform(lambda x: ','.join(x)).reset_index()
# abg_2[abg_2['patient_id'].str.contains(',')]

# %% [markdown]
# Dashboard

# %%
#define monk skin tone categories
mstlight = ['A','B','C','D']
mstmedium = ['E','F','G']
mstdark = ['H','I','J']

# %%
#begin creating the dashboard frame as db

#count number of unique patients per device
db = joined_updated[['device','patient_id', 'assigned_sex']].groupby(by=['device','patient_id']).count().reset_index().groupby(by='device').count().reset_index()
db = db.rename(columns={'patient_id':'Unique Subjects'})

#count assgined_sex
for i in ['Female','Male']:
    tdf = joined_updated[joined_updated['assigned_sex'] == i].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    db.rename(columns={'patient_id':i}, inplace=True)

########## count monk categories

#count those with monk dorsal that is light, medium, or dark
for i in [mstlight, mstmedium, mstdark]:
    # select only those with monk dorsal that is light, medium, or dark
    # groupby device and patient id to get each unique device-patient pair
    # then groupby device to get the count of unique patients per device
    tdf = joined_updated[joined_updated['monk_dorsal'].isin(i)].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
    # merge the new monk dorsal count data with the dashboard frame
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    db.rename(columns={'patient_id':'monk_dorsal_'+i[0]}, inplace=True)

# check if >= 1 in each of the 10 MST categories
# check the number of unique monk_dorsal per device
tdf = joined_updated.groupby(by=['device']).nunique()['monk_dorsal'].reset_index()
tdf.rename(columns={'monk_dorsal':'unique_monk_dorsal'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

########## count ITA categories
# number of patients with any ITA data
tdf = joined_updated[joined_updated['ita'].notnull()].groupby(by=['device','patient_id']).count().groupby('device').count()['record_id_x'].reset_index()
tdf.rename(columns={'record_id_x':'ita_any'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# ITA criteria: >=50, <=-45, >25, between 25 to -35, <-=35
itacriteria = [(joined_updated['ita']>50) & (joined_updated['monk_dorsal'].isin(['A','B','C','D'])), (joined_updated['ita']<=-45) & (joined_updated['monk_dorsal'].isin(['H','I','J'])), joined_updated['ita']>25, (joined_updated['ita']<25) & (joined_updated['ita']>-35), joined_updated['ita']<=-35]
criterianames = ['ita>=50&MonkABCD','ita<=-45&MonkHIJ','ita>25','ita25to-35','ita<=-35']

for i,j in zip(itacriteria,criterianames):
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
# check average number of samples per session
abg_updated_max = abg_updated.groupby(['patient_id','session']).max('sample').reset_index()[['patient_id','session','sample']]
abg_updated_max.rename(columns={'sample':'max_sample'}, inplace=True)
# merge the joined_updated table with the abg_updated_max table
joined_updated = joined_updated.merge(abg_updated_max, left_on=['patient_id','session'], right_on=['patient_id','session'], how='left')
# create a temp dataframe that has the average of 'sample_max' in each device
tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['max_sample'].reset_index()
tdf.rename(columns={'max_sample':'avg_sample'}, inplace=True)
tdf['avg_sample'] = tdf['avg_sample'].round(2)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# check range of number of samples per session
joined_updated['sample_range'] = joined_updated['max_sample'].apply(lambda x: 1 if (x >= 17) & (x <= 30) else 0)
tdf = joined_updated.groupby(by=['device']).min(numeric_only=True)['sample_range'].reset_index()
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# check if >= 90% of the sessions in the same device provide so2 data < 85
# create a new column called 'so2<85' that is 1 if so2 < 85, 0 if so2 >= 85 in abg_updated table
abg_updated_copy = abg_updated.copy()
abg_updated_copy['so2<85'] = abg_updated_copy['so2'].apply(lambda x: 1 if x < 85 else 0)
# group the abg_updated table by session, and calculate the mean of 'so2<85' in each session
abg_updated_copy['session'] = abg_updated_copy['session'].astype(str)
abg_updated_copy = abg_updated_copy.groupby(by=['session']).mean(numeric_only=True)['so2<85'].reset_index()
# change the so2<85 column to 1 if the column value is greater than 0, 0 otherwise
abg_updated_copy['so2<85'] = abg_updated_copy['so2<85'].apply(lambda x: 1 if x > 0 else 0)
# merge the abg_updated table with the joined_updated table
joined_updated['session'] = joined_updated['session'].astype(str)
joined_updated = joined_updated.merge(abg_updated_copy, left_on=['session'], right_on=['session'], how='left')
# group the so2<85 column by device, and calculate the mean of so2<85 in each device
tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['so2<85'].reset_index()
tdf['so2<85'] = tdf['so2<85'].apply(lambda x: x*100)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# check if >=70% participants/sessions provide data points in the 70%-80% decade (sao2)
abg_updated_copy = abg_updated.copy()
abg_updated_copy['sao2_70-80'] = abg_updated_copy['so2'].apply(lambda x: 1 if (x >= 70) & (x <= 80) else 0)
abg_updated_copy['session'] = abg_updated_copy['session'].astype(str)
abg_updated_copy = abg_updated_copy.groupby(by=['session']).mean(numeric_only=True)['sao2_70-80'].reset_index()
abg_updated_copy['sao2_70-80'] = abg_updated_copy['sao2_70-80'].apply(lambda x: 1 if x > 0 else 0)
joined_updated['session'] = joined_updated['session'].astype(str)
joined_updated = joined_updated.merge(abg_updated_copy, left_on=['session'], right_on=['session'], how='left')
tdf = joined_updated.groupby(by=['device']).mean(numeric_only=True)['sao2_70-80'].reset_index()
tdf['sao2_70-80'] = tdf['sao2_70-80'].apply(lambda x: x*100)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# %%
# Check if each decade between the 70% - 100% saturations contains 33% of the data points (sao2)
# group the joined_updated table by device, create three new columns called 'so2_70-80', 'so2_80-90', 'so2_90-100' that is the count of so2 in each decade
tdf = joined_updated.groupby(by=['device']).count()['so2'].reset_index()
tdf.rename(columns={'so2':'total'}, inplace=True)
tdf['so2_70-80'] = joined_updated[(joined_updated['so2'] <= 80) & (joined_updated['so2'] >= 70)].groupby(by=['device']).count()['so2'].reset_index()['so2']
tdf['so2_80-90'] = joined_updated[(joined_updated['so2'] >= 80) & (joined_updated['so2'] <= 90)].groupby(by=['device']).count()['so2'].reset_index()['so2']
tdf['so2_90-100'] = joined_updated[joined_updated['so2'] >= 90].groupby(by=['device']).count()['so2'].reset_index()['so2']
# calculate the percentage of each decade
tdf['so2_70-80'] = round(tdf['so2_70-80']/tdf['total'], 2) * 100
tdf['so2_80-90'] = round(tdf['so2_80-90']/tdf['total'], 2) * 100
tdf['so2_90-100'] = round(tdf['so2_90-100']/tdf['total'], 2) * 100
tdf = tdf.drop(columns=['total'])
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# group by device and count the number of sessions with >=25% of so2 data points in the 70%-80%, 80%-90%, and 90% above decade respectively
tdf = joined_updated.groupby(by=['device','session']).count()['so2'].reset_index()
tdf.rename(columns={'so2':'total'}, inplace=True)
tdf['so2_70-80'] = joined_updated[(joined_updated['so2'] <= 80) & (joined_updated['so2'] >= 70)].groupby(by=['device','session']).count()['so2'].reset_index()['so2']
tdf['so2_80-90'] = joined_updated[(joined_updated['so2'] >= 80) & (joined_updated['so2'] <= 90)].groupby(by=['device','session']).count()['so2'].reset_index()['so2']
tdf['so2_90-100'] = joined_updated[joined_updated['so2'] >= 90].groupby(by=['device','session']).count()['so2'].reset_index()['so2']
# calculate the percentage of each decade
tdf['so2_70-80'] = round(tdf['so2_70-80']/tdf['total'], 2) * 100
tdf['so2_80-90'] = round(tdf['so2_80-90']/tdf['total'], 2) * 100
tdf['so2_90-100'] = round(tdf['so2_90-100']/tdf['total'], 2) * 100
tdf = tdf.drop(columns=['total'])
# for each row in tdf, drop the row if any of the three decades is less than 25
tdf = tdf[(tdf['so2_70-80'] >= 25) & (tdf['so2_80-90'] >= 25) & (tdf['so2_90-100'] >= 25)]
# group by device and count the number of sessions in tdf
tdf = tdf.groupby(by=['device']).count()['session'].reset_index()
# rename session to '# of sessions with >=25% of so2 in the 3 decades' in tdf
tdf.rename(columns={'session':'session_count'}, inplace=True)
# merge db with tdf, if there is no session with >=25% of so2 in the 3 decades, fill the value with 0
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# %%
# fill zeroes
db.fillna(0, inplace=True)
# drop the assigned_sex column in db
db = db.drop(columns=['assigned_sex'])
#create a dictionary of column names and their descriptions
column_dict = {'device':'Device',
                'Unique Subjects':'Unique Subjects',
                'Female': 'Female',
                'Male': 'Male',
                'monk_dorsal_A':'Monk ABCD',
                'monk_dorsal_E':'Monk EFG',
                'monk_dorsal_H':'Monk HIJ',
                'unique_monk_dorsal':'Unique Monk',
                'ita_any':'Any ITA',
                'ita>=50&MonkABCD':'ITA >= 50 & Monk ABCD',
                'ita<=-45&MonkHIJ':'ITA <= -45 & Monk HIJ',
                'ita>25':'ITA > 25',
                'ita25to-35':'-35 < ITA <= 25',
                'ita<=-35':'ITA <= -35',
                'priority':'Test Priority',
                'avg_sample':'Avg Samples per Session',
                'sample_range':'17 <= Num Samples per Session <= 30',
                'so2<85':'%\n of Sessions Provides SaO2 < 85',
                'sao2_70-80':'%\n of Sessions Provides SaO2 in 70-80',
                'so2_70-80':'%\n of SaO2 in 70-80',
                'so2_80-90':'%\n of SaO2 in 80-90',
                'so2_90-100':'%\n of SaO2 in 90-100',
                'session_count':'# of Sessions with >=25%\n of SaO2 in 70-80, 80-90, 90-100'
                # 'mean_age_at_session':'Mean Age',
                # 'min_age_at_session':'Min Age',
                # 'max_age_at_session':'Max Age',
                # 'age_at_session_range':'Age Range',
                # 'mean_bmi':'Mean BMI',
                # 'min_bmi':'Min BMI',
                # 'max_bmi':'Max BMI',
                # 'bmi_range':'BMI Range',
                # 'priority':'Test priority'}
                }

# %% [markdown]
# - Criteria 1: Sample size >= 24 (unique patient_id)

# %% [markdown]
# - Criteria 2: Average of number of data points per participant/session = 24 (+/-4) (sao2)

# %% [markdown]
# - Criteria 3: Range of number of data points per participant/session in 17-30 (sao2)

# %% [markdown]
# Criteria 4: Each decade between the 70% - 100% saturations contains 33% of the data points (sao2)

# %% [markdown]
# - Criteria 5: >= 90% participants/sessions provide data points below 85% (sao2)

# %% [markdown]
# - Criteria 6: >=70% participants/sessions provide data points in the 70%-80% decade (sao2)

# %% [markdown]
# - Criteria 7: Each sex has approximately 40% percentage (assigned_sex)

# %% [markdown]
# Criteria 8: Number of repeated participants (patient_id, session)

# %% [markdown]
# - Criteria 9: >= 1 in each of the 10 MST categories

# %% [markdown]
# - Criteria 10: >= 25% in each of the following MST categories: 1-4, 5-7, 8-10

# %% [markdown]
# - Criteria 11: >= 25% in each of the following MST categories: 1-4(>25°), 5-7(>-35°, <=25°), 8-10(<=-35°)

# %% [markdown]
# - Criteria 12: >=1 subject in category MST 1-4 with ITA >=50°

# %% [markdown]
# - Criteria 13: >=2 subjects in category MST 8-10 with ITA <=-45°


