# %%
import pandas as pd
import numpy as np
import os
from redcap import Project
#from itables import show
import math

#from hypoxialab_functions import *

# %%
# functions

#load all data from redcap
def load_project(key):
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

    for index, row in df.iterrows(): #iterate through every patient
        for i in range(1,11): #iterate through each device in every patient
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
    mask = s > sums*threshold
    return ['background-color: green' if v else '' for v in mask]


# %%
try:
    is_streamlit == True
except NameError:
    is_streamlit = False

# %%
if is_streamlit == False:
    session = load_project('REDCAP_SESSION')
    session = session.reset_index()
    manual = load_project('REDCAP_MANUAL') 
    participant = load_project('REDCAP_PARTICIPANT')
    konica = load_project('REDCAP_KONICA')
    devices = load_project('REDCAP_DEVICES')
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
participant.drop(columns=['record_id'], inplace=True)
joined = joined.merge(participant.reset_index(), left_on='patient_id', right_on='subject_id', how='left')

# calculate age at session
joined['session_date']=pd.to_datetime(joined['session_date'])
joined['dob'] = pd.to_datetime(joined['dob'])

#age at session in years
joined['age_at_session'] = joined['session_date'].dt.year-joined['dob'].dt.year

#define monk skin tone categories
mstlight = ['A','B','C']
mstmedium = ['D','E','F','G']
mstdark = ['H','I','J']

# %%
#begin creatnig the dashboard frame as db
# count number of unique patients per device
db = joined[['device','patient_id']].groupby(by=['device','patient_id']).count().reset_index().groupby(by='device').count().reset_index()
db = db.rename(columns={'patient_id':'unique_patients'})

########## count monk categories

#count those with monk dorsal that is light, medium, or dark
for i in [mstlight, mstmedium, mstdark]:
    # select only those with monk dorsal that is light, medium, or dark
    # groupby device and patient id to get each unique device-patient pair
    # then groupby device to get the count of unique patients per device
    tdf = joined[joined['monk_dorsal'].isin(i)].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
    #merge the new monk dorsal count data with the dashboard frame
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    db.rename(columns={'patient_id':'monk_dorsal_'+i[0]}, inplace=True)


########## count ITA categories
# number of patients with any ITA data
tdf = joined[joined['ita'].notnull()].groupby(by=['device','patient_id']).count().groupby('device').count()['record_id'].reset_index()
tdf.rename(columns={'record_id':'ita_any'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# ITA criteria: >50, <-45, >25, between 25 to -35, <-35
itacriteria = [joined['ita']>50, joined['ita']<-45, joined['ita']>25, (joined['ita']<25) & (joined['ita']>-35), joined['ita']<-35]
criterianames = ['ita>50','ita<-45','ita>25','ita25to-35','ita<-35']

for i,j in zip(itacriteria,criterianames):
    # temp dataframe for each device, counting only the patients who meet the criteria
    tdf = joined[i].groupby(by=['device','patient_id']).count().reset_index().groupby(by='device').count().reset_index()
    # select only device and patient id columns, rename patient id to the criteria name
    tdf = tdf[['device','patient_id']].set_index('device').rename(columns={'patient_id':j})
    # merge with dashboard frame
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')


#### add age at session

db = db.merge(stats('age_at_session',joined), left_on='device', right_index=True, how='outer')
db = db.merge(stats('bmi',joined), left_on='device', right_index=True, how='outer')

# add device names and priority
## merge with devices, keeping all devices
db = devices[['manufacturer','model','priority']].merge(db, left_index=True, right_on='device', how='outer').reset_index().drop(columns=['index'])

# fill zeroes
db.fillna(0, inplace=True)


#create a dictionary of column names and their descriptions
column_dict = {'device':'Device',
                'unique_patients':'Unique Patients',
                'monk_dorsal_A':'Monk ABC',
                'monk_dorsal_D':'Monk DEF',
                'monk_dorsal_H':'Monk HIJ',
                'ita_any':'Any ITA',
                'ita>50':'ITA >50',
                'ita<-45':'ITA <-45',
                'ita>25':'ITA >25',
                'ita25to-35':'ITA 25 to -35',
                'ita<-35':'ITA <-35',
                'mean_age_at_session':'Mean Age',
                'min_age_at_session':'Min Age',
                'max_age_at_session':'Max Age',
                'age_at_session_range':'Age Range',
                'mean_bmi':'Mean BMI',
                'min_bmi':'Min BMI',
                'max_bmi':'Max BMI',
                'bmi_range':'BMI Range',
                'priority':'Test priority'}

#style monk columns with threshold of .25
db_style = (db.style
        .apply(highlight_value_greater,cols_to_sum=['monk_dorsal_A','monk_dorsal_D','monk_dorsal_H'], threshold=.25, subset=['monk_dorsal_A'])
        .apply(highlight_value_greater, cols_to_sum=['monk_dorsal_A','monk_dorsal_D','monk_dorsal_H'], threshold=.25, subset=['monk_dorsal_D'])
        .apply(highlight_value_greater, cols_to_sum=['monk_dorsal_A','monk_dorsal_D','monk_dorsal_H'], threshold=.25, subset=['monk_dorsal_H'])
        #now style column ita>25 with threshold of .25
        .apply(highlight_value_greater, cols_to_sum=['ita_any'], threshold=.25, subset=['ita>25'])
        # same with ita25to-35
        .apply(highlight_value_greater, cols_to_sum=['ita_any'], threshold=.25, subset=['ita25to-35'])
        # and same for ita<-35
        .apply(highlight_value_greater, cols_to_sum=['ita_any'], threshold=.25, subset=['ita<-35'])
        #highlight if number of patients with ITA<-45 >= 2 
        .map(lambda x: 'background-color: green' if x>=2 else "", subset=['ita<-45'])
        # apply formatting to all columns in column_dict
        .format(lambda x: f'{x:,.0f}', subset=list(column_dict.keys()))
)
db

# %%
haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk = pt_counts(konica,joined)

#print lenghts of each list in a loop 
#list descriptions
desc = ['patients with konica data', 'patients who have monk dorsal data', 'patients who have monk dorsal data and konica data', 'patients who have monk dorsal data but no konica data', 'patients who have konica data but no monk dorsal data']
for i,j in zip(desc,[haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk]):
    print(i,len(j))

# %%
# how many individual patients and their ITA ranges?
t1 = konica_unique_median.groupby(by=['upi']).agg({'ita':['min','max']}).reset_index()
#t1[t1['patient_id']==958].reset_index()
t1[t1['ita']['min']<-45]
# t1

# %% [markdown]
# # style

# %%
df = pd.DataFrame({
    'cost': [25.99, 97.45, 64.32, 14.78],
    'grams': [101.89, 20.924, 50.12, 40.015]
    })

# %%
# style so there is a border between cost and grams column
(df.style
    .set_table_styles([{'selector': 'th.col_heading',
                        'props': [('border-left', 'solid 1px black')]}])
    .format({'cost': '${0:,.2f}', 'grams': '{0:,.3f}'})
)

# %%
df = pd.DataFrame(np.random.randn(10, 4),
                  columns=['A', 'B', 'C', 'D'])

df.style.set_table_styles({
    'A': [{'selector': '',
           'props': [('color', 'red')]}],
    'B': [{'selector': 'td',
           'props': 'color: blue;'}]
}, overwrite=False)


