# %%
import pandas as pd
import numpy as np
import os
from redcap import Project
#from itables import show
import math

from hypoxialab_functions import *

# %%
session = load_project('REDCAP_SESSION')
session = session.reset_index()
manual = load_project('REDCAP_MANUAL') 
participant = load_project('REDCAP_PARTICIPANT')
konica = load_project('REDCAP_KONICA')
manual = reshape_manual(manual)

# keep only device and date columns from manual
manual = manual[['patient_id','device','date']]

# merge the session and manual dataframes, so one row is one session with one device. 10 devices in one session = 10 rows
joined = session.merge(manual, left_on=['patient_id','session_date'], right_on=['patient_id','date'], how='left')

# take the median of each unique session
konica_unique_median = konica.groupby(['date','upi','group']).median(numeric_only=True).reset_index()
# calculate the ITA
konica_unique_median['ita'] = konica_unique_median.apply(ita, axis=1)

# merge the konica data with the session data
joined = joined.merge(konica_unique_median, left_on=['patient_id','session_date'], right_on=['upi','date'], how='left')

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

#count those with monk fingernail that is light, medium, or dark
for i in [mstlight, mstmedium, mstdark]:
    # select only those with monk fingernail that is light, medium, or dark
    # groupby device and patient id to get each unique device-patient pair
    # then groupby device to get the count of unique patients per device
    tdf = joined[joined['monk_fingernail'].isin(i)].groupby(by=(['device','patient_id'])).count().reset_index().groupby('device').count()['patient_id']
    #merge the new monk fingernail count data with the dashboard frame
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')
    db.rename(columns={'patient_id':'monk_fingernail_'+i[0]}, inplace=True)


########## count ITA categories
# number of patients with any ITA data
tdf = joined[joined['ita'].notnull()].groupby(by=['device','patient_id']).count().groupby('device').count()['record_id'].reset_index()
tdf.rename(columns={'record_id':'ita_any'}, inplace=True)
db = db.merge(tdf, left_on='device', right_on='device', how='outer')

# ITA criteria: >50, <-45, >25, between 25 to -35, <-35
itacriteria = [joined['ita']>50, joined['ita']<-45, joined['ita']>25, (joined['ita']<25) & (joined['ita']>-35), joined['ita']<-35]
criterianames = ['ita>50','ita<-45','ita>25','ita25to-35','ita<-35']

for i,j in zip(itacriteria,criterianames):
    tdf = joined[i].groupby(by=['device','patient_id']).count().reset_index().groupby(by='device').count().reset_index()[['device','patient_id']].set_index('device').rename(columns={'patient_id':j})
    db = db.merge(tdf, left_on='device', right_on='device', how='outer')


#### add age at session

db = db.merge(stats('age_at_session',joined), left_on='device', right_index=True, how='outer')
db = db.merge(stats('bmi',joined), left_on='device', right_index=True, how='outer')

db

# %%
haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk = pt_counts(konica,joined)

#print lenghts of each list in a loop 
#list descriptions
desc = ['patients with konica data', 'patients who have monk fingernail data', 'patients who have monk fingernail data and konica data', 'patients who have monk fingernail data but no konica data', 'patients who have konica data but no monk fingernail data']
for i,j in zip(desc,[haskonica, hasmonk, hasboth, hasmonk_notkonica, haskonica_notmonk]):
    print(i,len(j))


