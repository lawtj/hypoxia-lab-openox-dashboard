import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import ast
from redcap import Project
import io
import os
import openox as ox
from session_functions import colormap


def getdf():
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['api_k']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='10', field='file')[0])
    df = pd.read_pickle(f)
    return df

df = getdf()

def get_qc_status():
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['REDCAP_QC']
    proj = Project(api_url, api_k)
    df = proj.export_records(format_type='df')

    return df

qc_status = get_qc_status()

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

def create_subset_frame(labview_samples, selected_session, show_cleaned=False):
    frame = labview_samples[labview_samples['session'] == selected_session]
    frame = frame.drop(columns=['session'])
    
    if frame['Nellcor PM1000N-1/SpO2'].sum() > 0:
        frame = frame.drop(columns=['Masimo 97/SpO2', 'Nellcor/SpO2', 'Nellcor/SpO2_diff_prev', 'Nellcor/SpO2_diff_next', 'Masimo 97/SpO2_diff_prev', 'Masimo 97/SpO2_diff_next'])
        frame = frame.rename(columns={
            'Nellcor PM1000N-1/SpO2': 'Nellcor/SpO2',
            'Nellcor PM1000N-1/SpO2_diff_prev': 'Nellcor/SpO2_diff_prev',
            'Nellcor PM1000N-1/SpO2_diff_next': 'Nellcor/SpO2_diff_next',
            'Rad97-60/SpO2': 'Masimo 97/SpO2',
            'Rad97-60/SpO2_diff_prev': 'Masimo 97/SpO2_diff_prev',
            'Rad97-60/SpO2_diff_next': 'Masimo 97/SpO2_diff_next'
        })
    else:
        frame = frame.drop(columns=['Masimo HB/SpO2', 'Masimo HB/SpO2_diff_prev', 'Masimo HB/SpO2_diff_next', 'Nellcor PM1000N-1/SpO2', 'Nellcor PM1000N-1/SpO2_diff_prev', 'Nellcor PM1000N-1/SpO2_diff_next', 'Rad97-60/SpO2', 'Rad97-60/SpO2_diff_prev', 'Rad97-60/SpO2_diff_next'])
    
    if not show_cleaned:
        # Add logic to set so2 and Nellcor/SpO2 to np.nan based on their stability
        frame['so2'] = frame.apply(lambda row: row['so2'] if row['so2_stable'] else np.nan, axis=1)
        frame['Nellcor/SpO2'] = frame.apply(lambda row: row['Nellcor/SpO2'] if row['Nellcor_stable'] else np.nan, axis=1)
        frame['bias'] = frame.apply(lambda row: row['bias'] if pd.notnull(row['so2']) and (pd.notnull(row['Nellcor/SpO2']) or pd.notnull(row['Nellcor PM1000N-1/SpO2'])) else np.nan, axis=1)
    
    criteria_check_tuple, criteria_check_df = ox.session_criteria_check(frame)
    
    # Create column 'col_so2_symbol' which is colormap['column'][1] if so2_stable is True, else 'cross'
    frame['so2_symbol'] = frame['so2_stable'].apply(lambda x: colormap['so2'][1] if x else 'cross')
    frame['Nellcor/SpO2_symbol'] = frame['Nellcor_stable'].apply(lambda x: colormap['Nellcor/SpO2'][1] if x else 'cross')
    
    # bias should be circle unless bias is greater than max_bias then it should be other
    frame['bias_symbol'] = 'circle'
    
    # ACTUALLY so2_line is always DarkSlateGrey
    frame['so2_line'] = 'DarkSlateGrey'
    
    # same for Nellcor/SpO2
    frame['Nellcor/SpO2_line'] = np.where((frame['Nellcor_stable'] == True) & (abs(frame['bias']) > max_bias), 'red', 'DarkSlateGrey')
    
    # bias line should be blue if bias is greater than max_bias
    frame['bias_line'] = np.where(abs(frame['bias']) > max_bias, 'blue', 'DarkSlateGrey')
    
    return frame, criteria_check_tuple, criteria_check_df

plotcolumns = ['so2', 'Nellcor/SpO2','bias']

def create_plot(frame, plotcolumns):
    fig = go.Figure()
    for column in plotcolumns:
        fig.add_trace(go.Scatter(
            x=frame['sample'], y=frame[column],
            mode='markers',
            name=column,
            marker=dict(
                symbol= frame[column + '_symbol'],
                color= colormap[column][0],
                size=12,
                opacity=0.8,
                line=dict(width=1.5, color=frame[column + '_line'])
            ),
        ))
    return fig


def update_qc_field(df, session_id, ):
    # first, check if qc_complete is True, but if any of the other fields are False, then qc_complete should be False
    # print each
    print(st.session_state.qc_notes, st.session_state.qc_missing, st.session_state.qc_date_discrepancy, st.session_state.qc_id_discrepancy, st.session_state.qc_quality, st.session_state.qc_complete)
    print(all([st.session_state.qc_notes, st.session_state.qc_missing, st.session_state.qc_date_discrepancy, st.session_state.qc_id_discrepancy, st.session_state.qc_quality]))
    if st.session_state.qc_complete and not all([st.session_state.qc_notes, st.session_state.qc_missing, st.session_state.qc_date_discrepancy, st.session_state.qc_id_discrepancy, st.session_state.qc_quality]):
        qc_message.error('Cannot set QC complete, some issues are not resolved.')
        return
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['REDCAP_QC']
    proj = Project(api_url, api_k)
    update_df = df[df['session_id'] == session_id]
    update_df['session_notes_addressed'] = st.session_state.qc_notes
    update_df['missing_files_resolved'] = st.session_state.qc_missing
    update_df['date_discrepancies_resolved'] = st.session_state.qc_date_discrepancy
    update_df['id_discrepancies_resolved'] = st.session_state.qc_id_discrepancy
    update_df['data_quality_checked'] = st.session_state.qc_quality
    update_df['qc_complete'] = st.session_state.qc_complete
    proj.import_records(update_df, import_format='df')
    qc_message.success('QC status updated', icon='✅')
    qc_message.caption(pd.Timestamp.now())
####################### title #######################

st.title('Session Quality Control')

df['missing_dates_tf'] = df['dates'].apply(lambda x: x['missing'])
df['missing_ptids'] = df['patient_ids'].apply(lambda x: x['missing_data'])
df['missing_ptids_tf'] = df['missing_ptids'].apply(lambda x: len(x)>0)
df['missing_abg_tf'] = df['missing_ptids'].apply(lambda x: 'Blood Gas' in x)
df['missing_konica_tf'] = df['missing_ptids'].apply(lambda x: 'Konica' in x)

####################### filters #######################

qc_complete_sessions_list = qc_status[qc_status['qc_complete'] == 1]['session_id'].tolist()
qc_incomplete_sessions_list = qc_status[qc_status['qc_complete'] == 0]['session_id'].tolist()

show_qc_status = st.selectbox('Select QC Status', ['Incomplete', 'Complete'])

left, right = st.columns(2)

with left:
    show_date_issues = st.checkbox('Show sessions with date discrepancies', value=False)
    show_missing_dates = st.checkbox('Show sessions with missing dates', value=False)
    show_ptid_issues = st.checkbox('Show sessions with patient ID discrepancies', value=True)

with right:
    show_missing_ptids = st.checkbox('Show sessions with missing patient IDs', value=False)
    show_missing_abgs = st.checkbox('Show sessions with missing ABG data', value=False)
    show_missing_konica = st.checkbox('Show sessions with missing Konica data', value=False)
dfstats = df.copy()

# Filtering logic
if not show_date_issues and not show_ptid_issues and not show_missing_dates and not show_missing_ptids and not show_missing_abgs and not show_missing_konica and show_qc_status == 'All':
    df = df  # Show all data if no filter is selected and QC status is 'All'
else:
    # Create masks based on user selection
    mask = pd.Series([True] * len(df))  # Initialize mask to True
    if show_date_issues:
        mask &= df['date_issues_tf']  # Update mask for date issues
    if show_ptid_issues:
        mask &= df['patient_id_issues_tf']  # Update mask for patient ID issues
    if show_missing_dates:
        mask &= df['missing_dates_tf']
    if show_missing_ptids:
        mask &= df['missing_ptids_tf']
    if show_missing_abgs:
        mask &= df['missing_abg_tf']
    if show_missing_konica:
        mask &= df['missing_konica_tf']
    
    # Apply QC status filter
    if show_qc_status == 'Complete':
        mask &= df['session_id'].isin(qc_complete_sessions_list)
    elif show_qc_status == 'Incomplete':
        mask &= df['session_id'].isin(qc_incomplete_sessions_list)

    df = df[mask]  # Apply the mask

selected_session = st.selectbox('Select Session ID', df['session_id'].sort_values(ascending=False), key='selected_session')


datesdict = df.loc[df['session_id'] == selected_session,'dates'].values[0]
ptids = df.loc[df['session_id'] == selected_session, 'patient_ids'].values[0]
###################### Initialize Session State #######################
st.session_state.qc_notes = qc_status.loc[qc_status['session_id'] == selected_session, 'session_notes_addressed'].values[0]
st.session_state.qc_missing = qc_status.loc[qc_status['session_id'] == selected_session, 'missing_files_resolved'].values[0]
st.session_state.qc_date_discrepancy = qc_status.loc[qc_status['session_id'] == selected_session, 'date_discrepancies_resolved'].values[0]
st.session_state.qc_id_discrepancy = qc_status.loc[qc_status['session_id'] == selected_session, 'id_discrepancies_resolved'].values[0]
st.session_state.qc_quality = qc_status.loc[qc_status['session_id'] == selected_session, 'data_quality_checked'].values[0]
st.session_state.qc_complete = qc_status.loc[qc_status['session_id'] == selected_session, 'qc_complete'].values[0]

####################### layout #######################

st.dataframe(df[['session_id','session_notes','date_issues_tf','patient_id_issues_tf','missing_abg_tf','missing_konica_tf','missing_ptids']].sort_index(ascending=False), use_container_width=True, column_config={
    "session_notes": st.column_config.TextColumn("Session Notes", width='medium'),
    "date_issues_tf": st.column_config.CheckboxColumn("Date Issues", width='small'),
    "patient_id_issues_tf": st.column_config.CheckboxColumn("Patient ID Issues", width='small'),
    'missing_ptids': st.column_config.ListColumn('Missing Patient IDs', width='medium'),
    'missing_abg_tf': st.column_config.CheckboxColumn("Missing ABG"),
    'missing_konica_tf':st.column_config.CheckboxColumn('Missing Konica')},hide_index=True)

with st.sidebar:
    max_bias = st.number_input('Highlight points where maximum bias is >= :', 0, 20, 10, 1)

    st.write('Total number of sessions:', len(dfstats))
    st.write('Number of sessions with date discrepancies:', len(dfstats[dfstats['date_issues_tf']]))
    st.write('Number of sessions with patient ID discrepancies:', len(dfstats[dfstats['patient_id_issues_tf']]))
    st.write('Number of sessions with missing ABG data:', len(dfstats[dfstats['missing_abg_tf']]))
    st.write('Number of sessions with missing Konica data:', len(dfstats[dfstats['missing_konica_tf']]))


st.markdown(f'## Session {selected_session}')
show_cleaned = st.checkbox('Show cleaned out points', value=False, key='show_cleaned')

with st.form(key='qc_form'):
    left, right = st.columns(2)
    with left:

        st.markdown('### Session Notes')
        st.checkbox('Session notes addressed', key='qc_notes')

        if pd.isna(df.loc[df['session_id']==selected_session,'session_notes'].values[0]):
            st.success('No notes found', icon="✅")
        else:
            st.info(df.loc[df['session_id']==selected_session,'session_notes'].values[0])
        

    with right:
        st.markdown('### Missing Files')
        st.checkbox('Missing files addressed', key='qc_missing')

        missing_abg = df.loc[df['session_id']==selected_session,'missing_abg_tf'].values[0]
        missing_konica = df.loc[df['session_id']==selected_session,'missing_konica_tf'].values[0]

        if missing_abg:
            st.error('Missing ABG file')
        if missing_konica:
            st.error('Missing Konica file')
        if not missing_abg and not missing_konica:
            st.success('No missing files found', icon="✅")

        
    left, right = st.columns(2)
    with left:
        st.markdown('### Date Discrepancies')
        st.checkbox('Date Discrepancies addressed', key='qc_date_discrepancy')

        # make a warning flag if there is date_issues_tf = True
        if df.loc[df['session_id']==selected_session, 'date_issues_tf'].values[0]:
            st.error('Date discrepancies found', icon="🚨")
        else:
            st.success('No date discrepancies found', icon="✅")
        st.markdown(f'''
        
        * **Session date:** {datesdict['dates']['session']} 

        * **Konica date:** {datesdict['dates']['konica']}

        * **ABG file date:** {datesdict['dates']['bloodgas']}

        * **Manual pulse ox entry date:** {datesdict['dates']['pulseox'] if datesdict['dates']['pulseox'] != [] else 'No manual pulse ox entry data'}
        ''')
    with right:
        st.markdown('### Patient ID Discrepancies')
        st.checkbox('Patient ID Discrepancies addressed', key='qc_id_discrepancy')

        # make a warning flag if there is patient_id_issues_tf = True
        if df.loc[df['session_id']==selected_session,'patient_id_issues_tf'].values[0]:
            st.error('Patient ID discrepancies found', icon="🚨")
        else:
            st.success('No patient ID discrepancies found', icon="✅")

        st.markdown(f'''
        * **Session patient ID**: {ptids['patient_ids']['session']} 

        * **Konica patient ID**: {ptids['patient_ids']['konica']}

        * **ABG file patient ID**: {ptids['patient_ids']['bloodgas']}

        * **Manual pulse ox entry patient ID**: {ptids['patient_ids']['pulseox']}
        ''')
    
    st.markdown('### Data Quality Check')
    st.checkbox('Data quality checked (no bad points)', key='qc_quality')
    st.write('ARMS calculations:')
    for value in df.loc[df['session_id'] == selected_session, 'arms'].values[0].items():
        st.write(value[0], round(value[1],2))

    st.plotly_chart(create_plot(create_subset_frame(labview_samples, selected_session, st.session_state.show_cleaned)[0], plotcolumns), use_container_width=True)
    st.dataframe(create_subset_frame(labview_samples, selected_session)[0].set_index('sample')
                 .drop(columns=['sample_diff_prev',
                                'sample_diff_next',
                                'Nellcor/SpO2_diff_prev',
                                'Nellcor/SpO2_diff_next',
                                'Masimo 97/SpO2_diff_prev',
                                'Masimo 97/SpO2_diff_next',
                                'so2_symbol',
                                'Nellcor/SpO2_symbol',
                                'bias_symbol',
                                'so2_line',
                                'Nellcor/SpO2_line',
                                'bias_line']), use_container_width=True)
    
    st.markdown('### Final QC')
    st.checkbox('QC complete', key='qc_complete')
    qc_message = st.container()
    
    submit_button = st.form_submit_button('Save QC', on_click=update_qc_field, args=(qc_status, selected_session))




