import streamlit as st
import hypoxialab_functions as ox
import plotly.express as px
import pandas as pd

### load data
SESSION_FORMS = ["hypoxia_lab_session", "sponsor", "study_pi", "device_settings"]  
session = ox.st_load_project('REDCAP_SESSION', forms=SESSION_FORMS) # to avoid key errors from pulling Labview Samples and Qc Status from redcap session
session = session.reset_index()
session = session.rename(columns={'record_id': 'session', 'patient_id': 'subject_id'})

konica = ox.load_project('REDCAP_KONICA')
konica = konica.rename(columns={'upi':'subject_id'})
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
konica['group'] = konica['group'].replace(ita_groups)
konica['ita'] = konica.apply(ox.ita, axis=1)
konica = konica[konica['ita'] == konica.groupby(['session', 'group'])['ita'].transform('median')]
konica.drop_duplicates(subset=konica.columns[1:], inplace=True)

### merge the datasets
session_konica = session.merge(konica, on=['subject_id','session'], how='inner')
session_konica['monk'] = session_konica.apply(ox.monkcolor, axis=1)
session_konica = session_konica[['subject_id','session', 'date', 'monk_forehead', 'group', 'monk', 'ita']]
session_konica = session_konica[session_konica['subject_id'].isin(session_konica.groupby('subject_id')['session'].nunique().reset_index().query('session > 1')['subject_id'])]
session_konica['monk_forehead'] = session_konica['monk_forehead'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
session_konica['monk_forehead_diff'] = session_konica.groupby('subject_id')['monk_forehead'].transform('max') - session_konica.groupby('subject_id')['monk_forehead'].transform('min')

st.write("### Count of repeated subjects in the lab: ", session_konica.groupby('subject_id')['session'].nunique().reset_index().query('session > 1')['subject_id'].nunique())
# st.write("Only subjects with monk_forehead change <= 1 letter are displayed below.")

### filters
with st.sidebar:
    st.write('### Filters')
    
    site_list = ['Dorsal', 'Forehead','Fingernail', 'Inner Upper Arm', 'Palmar']
    selected_site = st.selectbox('Anatomical Site', site_list)
    
    ITA_diff_list = ['>=30', '>=20', '>=10']
    selected_ITA_diff = st.selectbox('ITA Difference', ITA_diff_list)
    
    mst_diff_list = ['only <=1', 'all']
    selected_mst_diff = st.selectbox('Monk Forehead Change', mst_diff_list)
    
    # only look at repeated subjects that have monk_forehead change <= 1 and filter by selected site
    if selected_mst_diff == 'only <=1':
        session_konica = session_konica[(session_konica['monk_forehead_diff'] <= 1) & (session_konica['group'] == selected_site)]
    else:
        session_konica = session_konica[(session_konica['group'] == selected_site)]
    session_konica['ita_range'] = session_konica.groupby('subject_id')['ita'].transform('max') - session_konica.groupby('subject_id')['ita'].transform('min')
    session_konica['monk'] = session_konica['monk'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    session_konica['mst_range'] = session_konica.groupby('subject_id')['monk'].transform('max') - session_konica.groupby('subject_id')['monk'].transform('min')
    # filter by selected ita_diff threshold
    session_konica = session_konica[session_konica['ita_range'] >= int(selected_ITA_diff[2:])]
    # change monk back to letter
    session_konica['monk'] = session_konica['monk'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    session_konica = session_konica.drop(columns=['monk_forehead', 'group', 'monk_forehead_diff']).sort_values(by=['subject_id', 'ita']).reset_index(drop=True)

st.write(f"#### Count of subjects with ITA at {selected_site} changing by {selected_ITA_diff} degree: ", session_konica['subject_id'].nunique())

### scatter plot
ita_min_max = session_konica.groupby('subject_id')['ita'].agg(['min', 'max']).reset_index()
ita_min_max = pd.melt(ita_min_max, id_vars='subject_id', value_vars=['min', 'max'], var_name='min_max', value_name='ita')
ita_min_max = ita_min_max.merge(session_konica[['subject_id', 'ita', 'date']], on=['subject_id', 'ita'], how='inner')
ita_min_max['date'] = pd.to_datetime(ita_min_max['date'])
ita_min_max = ita_min_max.sort_values(by='date')
ita_min_max['date'] = pd.to_datetime(ita_min_max['date'].dt.strftime('%m-%Y'))
ita_vs_date = px.scatter(ita_min_max, x='date', y='ita', color='min_max', title='ITA vs Date by Min/Max ITA', 
                         labels={"min_max": "Min/Max ITA"}).update_xaxes(title_text='Date').update_yaxes(title_text='ITA', range=[-80, 80], dtick=20).update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(ita_vs_date, use_container_width=True)

### dataframe
selected_subject = st.selectbox(f'Select a subject from the {session_konica["subject_id"].nunique()} total', session_konica['subject_id'].unique())
session_konica = session_konica[session_konica['subject_id'] == selected_subject]
st.dataframe(session_konica.set_index('subject_id'), width=500)



