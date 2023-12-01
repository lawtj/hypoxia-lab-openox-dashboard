import os
from redcap import Project
import pandas as pd
import numpy as np
import streamlit as st
import math

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
    #patients who have monk fingernail data
    hasmonk = joined[joined['monk_fingernail'].notnull()]['patient_id'].unique().tolist()
    #patients who have monk fingernail data and konica data
    hasboth = list(set(haskonica) & set(hasmonk))
    #patients who have monk fingernail data but no konica data
    hasmonk_notkonica = list(set(hasmonk) - set(haskonica))
    #patients who have konica data but no monk fingernail data
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

def monkcolor(row):
    if pd.notna(row['group']):
        if 'Arm' in row['group']:
            return row['monk_upper_arm']
        elif 'Dorsal' in row['group']:
            return row['monk_dorsal']
        elif 'Forehead' in row['group']:
            return row['monk_forehead']
        elif 'Palmar' in row['group']:
            return row['monk_palmar']
        elif 'Fingernail' in row['group']:
            return row['monk_fingernail']
        
# # style database
# def highlight_value_greater(s, cols_to_sum,threshold):
#     sums = db[cols_to_sum].sum(axis=1)
#     mask = s > sums*threshold
#     return ['background-color: green' if v else '' for v in mask]
