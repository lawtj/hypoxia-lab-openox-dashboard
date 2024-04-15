import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
from redcap import Project
import io

def get_dict(x):
    try:
        # replace nan with np.nan in the string
        x = x.replace('nan', "'np.nan'")
        return ast.literal_eval(x)
    except:
        #print error
        return 'error' + str(x)

# df = pd.read_csv("qidf.csv", index_col='session_id')

@st.cache_data
def getdf():
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['api_k']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='10', field='file')[0])
    df = pd.read_csv(f, index_col='session_id')
    return df

df = getdf()

####################### title #######################

st.title('Session Quality Control')

####################### filters #######################

show_date_issues = st.checkbox('Show sessions with date issues', value=True)
show_ptid_issues = st.checkbox('Show sessions with patient ID issues', value=True)

if not show_date_issues and not show_ptid_issues:
    print('showing all data')
else:
    if show_date_issues and not show_ptid_issues:
        df = df[df['date_issues_tf']]
    elif not show_date_issues and show_ptid_issues:
        df = df[df['patient_id_issues_tf']]
    else:
        df = df[df['date_issues_tf'] | df['patient_id_issues_tf']]



selected_session = st.selectbox('Select Session ID', df.index)


datesdict = get_dict(df.loc[selected_session]['dates'])
ptids = get_dict(df.loc[selected_session]['patient_ids'])

####################### layout #######################


tab_overview, tab_details = st.tabs(['Overview', 'Details'])

with tab_overview:
    st.dataframe(df[['session_notes','date_issues_tf','patient_id_issues_tf']], use_container_width=True, column_config={
        "session_notes": st.column_config.TextColumn("Session Notes", width='large'),
        "date_issues_tf": st.column_config.CheckboxColumn("Date Issues", width='small'),
        "patient_id_issues_tf": st.column_config.CheckboxColumn("Patient ID Issues", width='small')})

with tab_details:

    st.markdown(f'## Session {selected_session}')

    st.markdown('### Notes')
    if pd.isna(df.loc[selected_session]['session_notes']):
        st.write('No notes found')
    else:
        st.write(df.loc[selected_session]['session_notes'])

    left, right = st.columns(2)

    with left:
        st.markdown('### Dates')
        # make a warning flag if there is date_issues_tf = True
        if df.loc[selected_session]['date_issues_tf']:
            st.error('Date issues found', icon="🚨")
        else:
            st.success('No date issues found', icon="✅")
        st.markdown(f'''
        
        * **Session date:** {datesdict['session']} 

        * **Konica date:** {datesdict['konica']}

        * **ABG file date:** {datesdict['bloodgas']}

        * **Manual pulse ox entry date:** {datesdict['pulseox'] if datesdict['pulseox'] != [] else 'No manual pulse ox entry data'}
        ''')

    with right:
        st.markdown('### Patient ID Issues')
        # make a warning flag if there is patient_id_issues_tf = True
        if df.loc[selected_session]['patient_id_issues_tf']:
            st.error('Patient ID issues found', icon="🚨")
        else:
            st.success('No patient ID issues found', icon="✅")

        st.markdown(f'''
        * **Session patient ID**: {ptids['session']} 

        * **Konica patient ID**: {ptids['konica']}

        * **ABG file patient ID**: {ptids['bloodgas']}

        * **Manual pulse ox entry patient ID**: {ptids['pulseox'] if 'pulseox' in ptids.keys() else 'No manual pulse ox entry data'}
        ''')

