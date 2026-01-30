import streamlit as st
from redcap import Project
import math
import pandas as pd
import numpy as np
import plotly.express as px
import hypoxialab_functions as ox

# @st.cache_data
SESSION_FORMS = ["hypoxia_lab_session", "sponsor", "study_pi", "device_settings"]  
session = ox.st_load_project('REDCAP_SESSION', forms=SESSION_FORMS) # to avoid key errors from pulling Labview Samples and Qc Status from redcap session
session = session.reset_index().rename(columns={'record_id':'session'})
konica = ox.st_load_project('REDCAP_KONICA').rename(columns={'upi':'patient_id'})
konica['ita'] = konica.apply(ox.ita, axis=1)
# konica = konica[konica['ita'] == konica.groupby(['session', 'group'])['ita'].transform('median')]
konica = konica.groupby(['session','group']).median(numeric_only=True).reset_index()

# Merge session and konica data
session_konica = pd.merge(session, konica, on=['patient_id','session'], how='outer')
# delete the 0 after the decimal point for session
session_konica['session'] = session_konica['session'].astype(str).str.replace('.0','')

ita_groups = {'Back Earlobe (F)': 'Back Earlobe',
'Back earlobe (F)': 'Back Earlobe',
'Cheek (I)': 'Cheek',
'Dorsal (B)': 'Dorsal',
'Dorsal - DIP (B)': 'Dorsal',
'Fingernail (A)': 'Fingernail',
'Forehead (E)': 'Forehead',
'Forehead (G)': 'Forehead',
'Front Earlobe (E)': 'Front Earlobe',
'Front earlobe (E)' : 'Front Earlobe',
'Inner Upper Arm (D)': 'Inner Upper Arm',
'Nare (H)': 'Nare',
'Palmar (C)': 'Palmar',
'Palmar - Opposite DIP (C)': 'Palmar'}
session_konica['group'] = session_konica['group'].map(ita_groups)

st.title('Session Information')
sorted_patient_ids = sorted(session_konica['patient_id'].unique())
selected_sid = st.selectbox('Select a patient_id', sorted_patient_ids)

st.write(f"## Patient ID: {selected_sid}")

tab1, tab2 = st.tabs(['Session Information', 'ITA by Group per Session'])
with tab1:
    st.write(f"Prior Session Information for Patient ID: {selected_sid}")
    st.table(session_konica[session_konica['patient_id'] == selected_sid][['session', 'session_date', 'monk_forehead', 'session_notes']].drop_duplicates())
with tab2:
    ita_by_group_per_session = px.scatter(session_konica[session_konica['patient_id'] == selected_sid], x='group', y='ita',hover_data=['session','ita'])
    st.plotly_chart(ita_by_group_per_session)

