import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
from redcap import Project
import io

@st.cache_data
def getdf():
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['api_k']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='10', field='file')[0])
    df = pd.read_pickle(f)
    return df

df = getdf()
dfstats = df.copy()

####################### title #######################

st.title('Session Quality Control')

df['missing_dates_tf'] = df['dates'].apply(lambda x: x['missing'])
df['missing_ptids'] = df['patient_ids'].apply(lambda x: x['missing_data'])
df['missing_ptids_tf'] = df['missing_ptids'].apply(lambda x: len(x)>0)

# st.dataframe(df, hide_index=True)

####################### filters #######################

show_date_issues = st.checkbox('Show sessions with date discrepancies', value=False)
show_missing_dates = st.checkbox('Show sessions with missing dates', value=False)
show_ptid_issues = st.checkbox('Show sessions with patient ID discrepancies', value=True)
show_missing_ptids = st.checkbox('Show sessions with missing patient IDs', value=False)

# Filtering logic
if not show_date_issues and not show_ptid_issues and not show_missing_dates and not show_missing_ptids:
    df = df  # Show all data if no filter is selected
else:
    # Create masks based on user selection
    mask = pd.Series([False] * len(df))  # Initialize mask to False
    if show_date_issues:
        mask |= df['date_issues_tf']  # Update mask for date issues
    if show_ptid_issues:
        mask |= df['patient_id_issues_tf']  # Update mask for patient ID issues
    if show_missing_dates:
        mask |= df['missing_dates_tf']
    if show_missing_ptids:
        mask |= df['missing_ptids_tf']
    

    df = df[mask]  # Apply the mask

selected_session = st.selectbox('Select Session ID', df['session_id'].sort_values(ascending=False))


datesdict = df.loc[df['session_id'] == selected_session,'dates'].values[0]
ptids = df.loc[df['session_id'] == selected_session, 'patient_ids'].values[0]

####################### layout #######################


tab_overview, tab_details = st.tabs(['Overview', 'Details'])

with tab_overview:
    st.dataframe(df[['session_id','session_notes','date_issues_tf','patient_id_issues_tf','missing_ptids']].sort_index(ascending=False), use_container_width=True, column_config={
        "session_notes": st.column_config.TextColumn("Session Notes", width='large'),
        "date_issues_tf": st.column_config.CheckboxColumn("Date Issues", width='small'),
        "patient_id_issues_tf": st.column_config.CheckboxColumn("Patient ID Issues", width='small'),
        'missing_ptids': st.column_config.ListColumn('Missing Patient IDs', width='medium')},hide_index=True)
    st.write('Total number of sessions:', len(dfstats))
    st.write('Number of sessions with date discrepancies:', len(dfstats[dfstats['date_issues_tf']]))
    st.write('Number of sessions with patient ID discrepancies:', len(dfstats[dfstats['patient_id_issues_tf']]))

with tab_details:

    st.markdown(f'## Session {selected_session}')

    st.markdown('### Notes')
    if pd.isna(df.loc[df['session_id']==selected_session,'session_notes'].values[0]):
        st.write('No notes found')
    else:
        st.write(df.loc[df['session_id']==selected_session,'session_notes'].values[0])

    left, right = st.columns(2)

    with left:
        st.markdown('### Dates')
        # make a warning flag if there is date_issues_tf = True
        if df.loc[df['session_id']==selected_session, 'date_issues_tf'].values[0]:
            st.error('Date issues found', icon="🚨")
        else:
            st.success('No date issues found', icon="✅")
        st.markdown(f'''
        
        * **Session date:** {datesdict['dates']['session']} 

        * **Konica date:** {datesdict['dates']['konica']}

        * **ABG file date:** {datesdict['dates']['bloodgas']}

        * **Manual pulse ox entry date:** {datesdict['dates']['pulseox'] if datesdict['dates']['pulseox'] != [] else 'No manual pulse ox entry data'}
        ''')

    with right:
        st.markdown('### Patient ID Issues')
        # make a warning flag if there is patient_id_issues_tf = True
        if df.loc[df['session_id']==selected_session,'patient_id_issues_tf'].values[0]:
            st.error('Patient ID issues found', icon="🚨")
        else:
            st.success('No patient ID issues found', icon="✅")

        st.markdown(f'''
        * **Session patient ID**: {ptids['patient_ids']['session']} 

        * **Konica patient ID**: {ptids['patient_ids']['konica']}

        * **ABG file patient ID**: {ptids['patient_ids']['bloodgas']}

        * **Manual pulse ox entry patient ID**: {ptids['patient_ids']['pulseox']}
        ''')

