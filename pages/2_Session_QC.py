import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import ast
import io
import os
import openox as ox
from session_functions import colormap
from datetime import datetime
import re
from libsql_client import Client
import asyncio


st.set_page_config(page_title='Session Quality Control', layout='wide')


# --- Turso Client Initialization ---
@st.cache_resource
def get_turso_client():
    """Initializes and returns a Turso database client."""
    url = st.secrets.get("TURSO_DB_URL")
    auth_token = st.secrets.get("TURSO_AUTH_TOKEN")
    if not url or not auth_token:
        raise ValueError("Turso database URL and auth token must be set in secrets.")

    if not url.startswith("libsql://"):
        url = "libsql://" + url.replace("https://", "").replace("http://", "")

    return Client(url, auth_token=auth_token)


# this df is the automated QC check that runs with labview_samples
# it is the output of OpenOxQI.ipynb
@st.cache_data(ttl='1h')
def getdf():
    """Fetches automated QC data from Turso DB."""
    client = get_turso_client()
    rs = asyncio.run(client.execute("SELECT * FROM automated_qc"))
    df = rs.to_pandas()

    # Convert columns back to their original types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'True': True, 'False': False, 'nan': np.nan, 'None': None})

    # Handle columns that are dicts/lists
    for col in ['dates', 'patient_ids', 'arms', 'abg_timestamps', 'konica_timestamps', 'manual_pulse_ox_timestamps']:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)

    return df


automated_qc_df = getdf()

#QC status is the manual QC review
@st.cache_data(ttl='1h')
def get_qc_status():
    """Fetches QC status data from Turso DB."""
    client = get_turso_client()
    rs = asyncio.run(client.execute("SELECT * FROM qc_status"))
    df = rs.to_pandas()
    # Convert types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'True': True, 'False': False, 'nan': np.nan, 'None': None})
    return df

qc_status = get_qc_status()

def label_manual_samples(labview_samples, drop_dict):
    # label samples to be dropped
    for session, samples in drop_dict.items():
        labview_samples.loc[(labview_samples['session'] == session), 'manual_so2'] = 'keep'
        labview_samples.loc[(labview_samples['session'] == session) & (labview_samples['sample'].isin(samples)), 'manual_so2'] = 'reject'
    # now compare the manual_so2 column with the algo column
    labview_samples['manual_algo_compare'] = None
    # start with the samples that were rejected by the manual_so2 column but kept by the algo column
    labview_samples.loc[(labview_samples['manual_so2'] == 'reject') & (labview_samples['so2_stable'] == True),'manual_algo_compare'] = 'manual reject'
    # now the samples that were kept by the manual_so2 column but rejected by the algo column
    labview_samples.loc[(labview_samples['manual_so2'] == 'keep') & (labview_samples['so2_stable'] == False),'manual_algo_compare'] = 'manual keep'
    # label keep both
    labview_samples.loc[(labview_samples['manual_so2'] == 'keep') & (labview_samples['so2_stable'] == True),'manual_algo_compare'] = 'keep (both)'
    # label reject both
    labview_samples.loc[(labview_samples['manual_so2'] == 'reject') & (labview_samples['so2_stable'] == False),'manual_algo_compare'] = 'reject (both)'
    return labview_samples

@st.cache_data(ttl='1h')
def get_labview_samples():
    """Fetches labview samples data from Turso DB."""
    client = get_turso_client()
    rs = asyncio.run(client.execute("SELECT * FROM labview_samples"))
    labview_samples = rs.to_pandas()

    # Convert types
    for col in labview_samples.columns:
        labview_samples[col] = pd.to_numeric(labview_samples[col], errors='ignore')
        if labview_samples[col].dtype == 'object':
            labview_samples[col] = labview_samples[col].replace({'True': True, 'False': False, 'nan': np.nan, 'None': None})

    from exclude_unclean import drop_dict
    labview_samples = label_manual_samples(labview_samples, drop_dict)
    return labview_samples

labview_samples = get_labview_samples()

# Function to safely evaluate the string into a dictionary
def safe_literal_eval(val):
    if val == '{}':
        return {}
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

armsdict = {row['session_id']: safe_literal_eval(row['arms']) for index, row in automated_qc_df.iterrows() if pd.notnull(row['arms'])}

def create_subset_frame(labview_samples, selected_session, show_cleaned=False):
    frame = labview_samples[labview_samples['session'] == selected_session]
    frame = frame.drop(columns=['session'])
    
    if frame['Nellcor PM1000N-1/SpO2'].sum() > 0:
        frame = frame.drop(columns=['Masimo 97/SpO2', 'Nellcor/SpO2', 'Masimo 97/PI', 'Nellcor/PI', 'Nellcor/SpO2_diff_prev', 'Nellcor/SpO2_diff_next', 'Masimo 97/SpO2_diff_prev', 'Masimo 97/SpO2_diff_next'])
        frame = frame.rename(columns={
            'Nellcor PM1000N-1/SpO2': 'Nellcor/SpO2',
            'Nellcor PM1000N-1/PI': 'Nellcor/PI',
            'Nellcor PM1000N-1/SpO2_diff_prev': 'Nellcor/SpO2_diff_prev',
            'Nellcor PM1000N-1/SpO2_diff_next': 'Nellcor/SpO2_diff_next',
            'Rad97-60/SpO2': 'Masimo 97/SpO2',
            'Rad97-60/PI': 'Masimo 97/PI',
            'Rad97-60/SpO2_diff_prev': 'Masimo 97/SpO2_diff_prev',
            'Rad97-60/SpO2_diff_next': 'Masimo 97/SpO2_diff_next'
        })
    else:
        frame = frame.drop(columns=['Nellcor PM1000N-1/SpO2', 'Nellcor PM1000N-1/PI','Nellcor PM1000N-1/SpO2_diff_prev', 'Nellcor PM1000N-1/SpO2_diff_next', 'Rad97-60/SpO2', 'Rad97-60/PI', 'Rad97-60/SpO2_diff_prev', 'Rad97-60/SpO2_diff_next'])
    
    if not show_cleaned:
        # Add logic to set so2 and Nellcor/SpO2 to np.nan based on their stability
        frame['so2'] = frame.apply(lambda row: row['so2'] if row['so2_stable'] else np.nan, axis=1)
        frame['Nellcor/SpO2'] = frame.apply(lambda row: row['Nellcor/SpO2'] if row['Nellcor_stable'] else np.nan, axis=1)
        frame['Masimo 97/SpO2'] = frame.apply(lambda row: row['Masimo 97/SpO2'] if row['Masimo_stable'] else np.nan, axis=1)
        frame['bias'] = frame.apply(lambda row: row['bias'] if pd.notnull(row['so2']) and (pd.notnull(row['Nellcor/SpO2']) or ('Nellcor PM1000N-1/SpO2' in row and pd.notnull(row['Nellcor PM1000N-1/SpO2']))) else np.nan, axis=1)
    
    criteria_check_tuple, criteria_check_df = ox.session_criteria_check(frame)
    
    # Create column 'col_so2_symbol' which is colormap['column'][1] if so2_stable is True, else 'cross'
    frame['so2_symbol'] = frame['so2_stable'].apply(lambda x: colormap['so2'][1] if x else 'cross')
    frame['Nellcor/SpO2_symbol'] = frame['Nellcor_stable'].apply(lambda x: colormap['Nellcor/SpO2'][1] if x else 'cross')
    frame['Masimo 97/SpO2_symbol'] = frame['Masimo_stable'].apply(lambda x: colormap['Masimo 97/SpO2'][1] if x else 'cross')
    
    # bias should be circle unless bias is greater than max_bias then it should be other
    frame['bias_symbol'] = 'circle'
    
    # ACTUALLY so2_line is always DarkSlateGrey
    frame['so2_line'] = 'DarkSlateGrey'
    
    # same for Nellcor/SpO2
    frame['Nellcor/SpO2_line'] = np.where((frame['Nellcor_stable'] == True) & (abs(frame['bias']) > max_bias), 'red', 'DarkSlateGrey')

    # same for Masimo 97/SpO2
    frame['Masimo 97/SpO2_line'] = np.where((frame['Masimo_stable'] == True) & (abs(frame['bias']) > max_bias), 'red', 'DarkSlateGrey')
    
    # bias line should be blue if bias is greater than max_bias
    frame['bias_line'] = np.where(abs(frame['bias']) > max_bias, 'blue', 'DarkSlateGrey')
    
    return frame, criteria_check_tuple, criteria_check_df


def create_plot(frame, plotcolumns, limit_to_manual_sessions=False):
    fig = go.Figure()
    for column in plotcolumns:
        visiblestatus = True if column in ['so2', 'Nellcor/SpO2'] else 'legendonly'
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
            ), visible=visiblestatus,
        ))
    if limit_to_manual_sessions:
        # add text labeling points for manual_algo_compare
        for index, row in frame.iterrows():
            if row['manual_algo_compare'] == 'manual reject':
                fig.add_annotation(x=row['sample'], y=row['so2']+10, text='manual reject', showarrow=True)
            if row['manual_algo_compare'] == 'manual keep':
                fig.add_annotation(x=row['sample'], y=row['so2'], text='manual keep', showarrow=True)
    return fig


def on_click_next():
    if next_session is not None:
        st.session_state['selected_session'] = next_session
    else:
        st.session_state['selected_session'] = previous_session
def on_click_previous():
    if previous_session is not None:
        st.session_state['selected_session'] = previous_session
    else:
        st.session_state['selected_session'] = next_session

async def update_qc_field_async(df, session_id):
    # first, check if qc_complete is True, but if any of the other fields are False, then qc_complete should be False
    if st.session_state.qc_complete and not all([st.session_state.qc_notes, st.session_state.qc_missing, st.session_state.qc_date_discrepancy, st.session_state.qc_id_discrepancy, st.session_state.qc_quality]):
        qc_message.error('Cannot set QC complete, some issues are not resolved.')
        return

    client = get_turso_client()

    # Construct the SET part of the UPDATE statement
    set_clauses = [
        "session_notes_addressed = ?",
        "missing_files_resolved = ?",
        "date_discrepancies_resolved = ?",
        "id_discrepancies_resolved = ?",
        "data_quality_checked = ?",
        "qc_complete = ?",
        "data_quality_notes = ?",
        "data_quality_action = ?"
    ]

    # SQLite expects boolean values as 0 or 1
    params = [
        int(st.session_state.qc_notes),
        int(st.session_state.qc_missing),
        int(st.session_state.qc_date_discrepancy),
        int(st.session_state.qc_id_discrepancy),
        int(st.session_state.qc_quality),
        int(st.session_state.qc_complete),
        st.session_state.qc_quality_notes,
        int(st.session_state.qc_data_qualtiy_action)
    ]

    sql = f"UPDATE qc_status SET {', '.join(set_clauses)} WHERE session_id = ?"
    params.append(str(session_id)) # session_id is stored as text

    await client.execute(sql, params)

    qc_message.success(f'Session {selected_session}: QC status updated in Turso DB', icon='✅')
    qc_message.caption(pd.Timestamp.now())
    # Re-run the query to get the updated data
    st.cache_data.clear()
    on_click_next()

def update_qc_field(df, session_id):
    asyncio.run(update_qc_field_async(df, session_id))

####################### title #######################

st.title('Session Quality Control')

automated_qc_df['missing_dates_tf'] = automated_qc_df['dates'].apply(lambda x: x['missing'])
automated_qc_df['missing_ptids'] = automated_qc_df['patient_ids'].apply(lambda x: x['missing_data'])
automated_qc_df['missing_ptids_tf'] = automated_qc_df['missing_ptids'].apply(lambda x: len(x)>0)
automated_qc_df['missing_abg_tf'] = automated_qc_df['missing_ptids'].apply(lambda x: 'Blood Gas' in x)
automated_qc_df['missing_konica_tf'] = automated_qc_df['missing_ptids'].apply(lambda x: 'Konica' in x)
automated_qc_df['missing_labview_tf'] = automated_qc_df['missing_ptids'].apply(lambda x: 'Labview' in x)

####################### filters #######################

qc_complete_sessions_list = qc_status[qc_status['qc_complete'] == 1]['session_id'].tolist()
qc_incomplete_sessions_list = qc_status[qc_status['qc_complete'] == 0]['session_id'].tolist()

show_qc_status = st.selectbox('Select QC Status', ['Incomplete', 'Complete'])

qcstats = qc_status.copy()
combine_logic = st.toggle('Show matches that meet ANY of the filter criteria', value=True)

left, right = st.columns(2)
with left:
    show_session_note_issues = st.toggle('Show sessions with session note issues', value=True)
    show_missing_file_issues = st.toggle('Show sessions with missing files', value=True)
    show_date_issues = st.toggle('Show sessions with date discrepancies', value=True)
with right:
    show_ptid_issues = st.toggle('Show sessions with patient ID discrepancies', value=True)
    show_data_quality_issues = st.toggle('Show sessions with data quality issues', value=True)
    show_data_quality_action = st.toggle('Limit to sessions that require data action/further data review', value=False)

with st.sidebar:

    filter_by_bias = st.toggle('Filter by bias', value=False)
    if filter_by_bias:
        st.write('Show sessions where bias is:')
        left, right = st.columns(2)
        with left:
            bias_comparison = st.selectbox('label', ['>=', '<='], label_visibility='collapsed')
        with right:
            max_bias = st.number_input('collapsed', 0, 20, 10, 1, label_visibility='collapsed')
    else:
        max_bias = 10
        bias_comparison = '>='  # Default to a non-effective comparison

    filter_by_nellcor_arms = st.toggle('Filter by Nellcor ARMS', value=False)
    if filter_by_nellcor_arms:
        st.write('Show sessions where Nellcor ARMS is:')
        left, right = st.columns(2)
        with left:
            arms_comparison = st.selectbox('Select comparison for Nellcor ARMS:', ['>=', '<='], label_visibility='collapsed')
        with right:
            max_arms = st.number_input('Show sessions where Nellcor ARMS is:', 0, 20, 3, 1, label_visibility='collapsed')
    else:
        max_arms = 3
        arms_comparison = '<='  # Default to a non-effective comparison

    show_cleaned = st.toggle('Show cleaned out points', value=False, key='show_cleaned')

    limit_to_manual_sessions = st.toggle('Limit to manual sessions', value=False)

    if limit_to_manual_sessions:
        #calculate stats on sessions that were manually reviewed
        manual_stats_df = labview_samples[labview_samples['manual_so2'].notnull()]
        st.write(manual_stats_df['manual_algo_compare'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    

    qcstats_summary = {
    "Metric": [
        "Total number of sessions",
        "Number of sessions with session note issues",
        "Number of sessions with missing files",
        "Number of sessions with date discrepancies",
        "Number of sessions with patient ID discrepancies",
        "Number of sessions with data quality issues",
        "Number of sessions with QC incomplete",
        "Number of sessions with QC complete"
    ],
    "Count": [
        len(qcstats),
        len(qcstats[qcstats['session_notes_addressed'] == False]),
        len(qcstats[qcstats['missing_files_resolved'] == False]),
        len(qcstats[qcstats['date_discrepancies_resolved'] == False]),
        len(qcstats[qcstats['id_discrepancies_resolved'] == False]),
        len(qcstats[qcstats['data_quality_checked'] == False]),
        len(qcstats[qcstats['qc_complete'] == 0]),
        len(qcstats[qcstats['qc_complete'] == 1])
    ]}
    # Convert the summary to a DataFrame
    qcstats_summary_df = pd.DataFrame(qcstats_summary).set_index("Metric")

    # Display the summary as a table
    st.table(qcstats_summary_df)


# Filtering logic based on qc status

if combine_logic:  # OR logic
    mask = pd.Series([False] * len(qc_status))
    if show_session_note_issues:
        mask |= qc_status['session_notes_addressed'] == False
    if show_missing_file_issues:
        mask |= qc_status['missing_files_resolved'] == False
    if show_date_issues:
        mask |= qc_status['date_discrepancies_resolved'] == False
    if show_ptid_issues:
        mask |= qc_status['id_discrepancies_resolved'] == False
    if show_data_quality_issues:
        mask |= qc_status['data_quality_checked'] == False
    if show_data_quality_action:
        mask &= qc_status['data_quality_action'] == True
    else:
        mask &= qc_status['data_quality_action'] == False

    if show_qc_status == 'Complete':
        mask &= qc_status['qc_complete'] == 1
    elif show_qc_status == 'Incomplete':
        mask &= qc_status['qc_complete'] == 0
else:  # AND logic
    mask = pd.Series([True] * len(qc_status))
    if show_session_note_issues:
        mask &= qc_status['session_notes_addressed'] == False
    else:
        mask &= qc_status['session_notes_addressed'] == True
    if show_missing_file_issues:
        mask &= qc_status['missing_files_resolved'] == False
    else:
        mask &= qc_status['missing_files_resolved'] == True
    if show_date_issues:
        mask &= qc_status['date_discrepancies_resolved'] == False
    else:
        mask &= qc_status['date_discrepancies_resolved'] == True
    if show_ptid_issues:
        mask &= qc_status['id_discrepancies_resolved'] == False
    else:
        mask &= qc_status['id_discrepancies_resolved'] == True
    if show_data_quality_issues:
        mask &= qc_status['data_quality_checked'] == False
    else:
        mask &= qc_status['data_quality_checked'] == True
    if show_data_quality_action:
        mask &= qc_status['data_quality_action'] == True
    else:
        mask &= qc_status['data_quality_action'] == False
    if show_qc_status == 'Complete':
        mask &= qc_status['qc_complete'] == 1
    elif show_qc_status == 'Incomplete':
        mask &= qc_status['qc_complete'] == 0

qc_status = qc_status[mask]

# Updated Filter Logic
if filter_by_bias:
    max_bias_list = labview_samples[(labview_samples['so2_stable']==True) & (labview_samples['Nellcor_stable']==True)][['session','bias']].assign(abs_bias=lambda x: x['bias'].abs()).groupby('session').max()
    if bias_comparison == '>=':
        biased_sessions = max_bias_list[max_bias_list['abs_bias'] >= max_bias].index.tolist()
    else:
        biased_sessions = max_bias_list[max_bias_list['abs_bias'] <= max_bias].index.tolist()

    qc_status = qc_status[qc_status['session_id'].isin(biased_sessions)]


if filter_by_nellcor_arms:
    if arms_comparison == '>=':
        high_arms_sessions = [key for key, value in armsdict.items() if 'Nellcor/SpO2' in value and value['Nellcor/SpO2'] >= max_arms]
        #add check for Nellcor PM1000N-1/SpO2 column
        high_arms_sessions += [key for key, value in armsdict.items() if 'Nellcor PM1000N-1/SpO2' in value and value['Nellcor PM1000N-1/SpO2'] >= max_arms]
    else:
        high_arms_sessions = [key for key, value in armsdict.items() if 'Nellcor/SpO2' in value and value['Nellcor/SpO2'] <= max_arms]
    qc_status = qc_status[qc_status['session_id'].isin(high_arms_sessions)]

if limit_to_manual_sessions:
    set1 = set(qc_status['session_id'].unique().tolist())
    set2 = set(labview_samples[labview_samples['manual_so2'] == 'reject']['session'].unique().tolist())
    manualsessionlist = list(set1.intersection(set2))
    qc_status = qc_status[qc_status['session_id'].isin(manualsessionlist)]

# dfm is the subset of the automated qc df that is in qc_status
dfm = automated_qc_df[automated_qc_df['session_id'].isin(qc_status['session_id'])][['session_id','session_notes','missing_abg_tf','missing_konica_tf','missing_labview_tf']]
#invert the true false values for missing files
dfm['missing_abg_tf'] = ~dfm['missing_abg_tf']
dfm['missing_konica_tf'] = ~dfm['missing_konica_tf']
dfm['missing_labview_tf'] = ~dfm['missing_labview_tf']

qc_display = qc_status.merge(dfm, on='session_id', how='left')

st.dataframe(qc_display[['session_id','session_notes','session_notes_addressed','missing_abg_tf','missing_konica_tf','missing_labview_tf','date_discrepancies_resolved','id_discrepancies_resolved','data_quality_checked','data_quality_action','data_quality_notes','qc_complete']].sort_index(ascending=False), use_container_width=True, column_config={
    "session_notes_addressed": st.column_config.CheckboxColumn("Session Notes", width='small'),
    'missing_abg_tf': st.column_config.CheckboxColumn("ABG file resolved", width='small'),
    'missing_konica_tf': st.column_config.CheckboxColumn("Konica file resolved", width='small'),
    'missing_labview_tf': st.column_config.CheckboxColumn("Labview file resolved", width='small'),
    "missing_files_resolved": st.column_config.CheckboxColumn("Missing Files", width='small'),
    "date_discrepancies_resolved": st.column_config.CheckboxColumn("Date Discrepancies", width='small'),
    "id_discrepancies_resolved": st.column_config.CheckboxColumn("Patient ID Discrepancies", width='small'),
    "data_quality_checked": st.column_config.CheckboxColumn("Data Quality", width='small'),
    "data_quality_notes": st.column_config.TextColumn("Data Quality Notes", width='medium'),
    "data_quality_action": st.column_config.CheckboxColumn("Requires Data Action", width='small'),
    "qc_complete": st.column_config.CheckboxColumn("QC Complete", width='small')}, hide_index=True)

st.write('Number of sessions:', len(qc_status))



#### navigation buttons

left, middle, right = st.columns(3)
session_list = qc_status['session_id'].sort_values(ascending=True).tolist()

with left:
    selected_session = st.selectbox('Select Session ID', session_list, key='selected_session')

    if pd.notnull(selected_session):
        index = session_list.index(selected_session)
        next_session = session_list[index+1] if index < len(session_list)-1 else None
        previous_session = session_list[index-1] if index > 0 else None

        with right:
            if next_session:
                next_button = st.button('Next session', key='next', on_click=on_click_next)
            else:
                st.write('No next session')
            if previous_session:
                previous_button = st.button('Previous session', key='previous', on_click=on_click_previous)
            else:
                st.write('No previous session')
    

##### 

if pd.notnull(selected_session):
    datesdict = automated_qc_df.loc[automated_qc_df['session_id'] == selected_session,'dates'].values[0]
    ptids = automated_qc_df.loc[automated_qc_df['session_id'] == selected_session, 'patient_ids'].values[0]
    try:
        frame = create_subset_frame(labview_samples, selected_session, st.session_state.show_cleaned)[0]
    except:
        frame = None

    abg_text = automated_qc_df.query('session_id == @selected_session')['abg2_timestamp'].values[0]
    # st.write(automated_qc_df.query('session_id == @selected_session'))
    if automated_qc_df.query('session_id == @selected_session')['abg2_timestamp'].values[0] is not None:
        match = re.search(r'Sample (\d*): (.*)', automated_qc_df.query('session_id == @selected_session')['abg2_timestamp'].values[0])
        abg_timestamp = pd.to_datetime(match.group(2)) if match else None
        abg_timestamp_num = pd.to_numeric(match.group(1)) if match else None
    else:
        abg_timestamp = None
        abg_timestamp_num = None
    

    try:
        labview_timestamp = pd.to_datetime(frame.loc[frame['sample'] == abg_timestamp_num, 'Timestamp'].values[0])
    except:
        labview_timestamp = None
    ###################### Initialize Session State #######################
    st.session_state.qc_notes = qc_status.loc[qc_status['session_id'] == selected_session, 'session_notes_addressed'].values[0]
    st.session_state.qc_missing = qc_status.loc[qc_status['session_id'] == selected_session, 'missing_files_resolved'].values[0]
    st.session_state.qc_date_discrepancy = qc_status.loc[qc_status['session_id'] == selected_session, 'date_discrepancies_resolved'].values[0]
    st.session_state.qc_id_discrepancy = qc_status.loc[qc_status['session_id'] == selected_session, 'id_discrepancies_resolved'].values[0]
    st.session_state.qc_quality = qc_status.loc[qc_status['session_id'] == selected_session, 'data_quality_checked'].values[0]
    st.session_state.qc_data_qualtiy_action = qc_status.loc[qc_status['session_id'] == selected_session, 'data_quality_action'].values[0]
    # check if nan and set to empty string otherwise set to value
    if pd.isna(qc_status.loc[qc_status['session_id'] == selected_session, 'data_quality_notes'].values[0]):
        st.session_state.qc_quality_notes = ''
    else:
        st.session_state.qc_quality_notes = qc_status.loc[qc_status['session_id'] == selected_session, 'data_quality_notes'].values[0]
    st.session_state.qc_complete = qc_status.loc[qc_status['session_id'] == selected_session, 'qc_complete'].values[0]

    ####################### layout #######################

    st.markdown(f'## Session {selected_session}')

    with st.form(key='qc_form'):
        left, right = st.columns(2)
        with left:

            st.markdown('### Session Notes')
            st.checkbox('Session notes addressed', key='qc_notes')

            if pd.isna(automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'session_notes'].values[0]):
                st.success('No notes found', icon="✅")
            else:
                st.info(automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'session_notes'].values[0])
            

        with right:
            st.markdown('### Missing Files')
            st.checkbox('Missing files addressed', key='qc_missing')

            missing_abg = automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'missing_abg_tf'].values[0]
            missing_konica = automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'missing_konica_tf'].values[0]
            missing_labview = automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'missing_labview_tf'].values[0]

            if missing_abg:
                st.error('Missing ABG file')
            if missing_konica:
                st.error('Missing Konica file')
            if missing_labview:
                st.error('Missing Labview file')
            if not missing_abg and not missing_konica and not missing_labview:
                st.success('No missing files found', icon="✅")

        import re
        left, right = st.columns(2)
        with left:
            st.markdown('### Date Discrepancies')
            st.checkbox('Date Discrepancies addressed', key='qc_date_discrepancy')

            # make a warning flag if there is date_issues_tf = True
            if automated_qc_df.loc[automated_qc_df['session_id']==selected_session, 'date_issues_tf'].values[0]:
                st.error('Date discrepancies found', icon="🚨")
            else:
                st.success('No date discrepancies found', icon="✅")
                # Create a DataFrame for the dates
            # Create a DataFrame for the dates
            dates_data = {
                'Label': ['Session date', f'Labview date (Sample {abg_timestamp_num})', f'ABG file date (Sample {abg_timestamp_num})',' Konica date',  'Manual pulse ox entry date', 
                        ],
                'Date': [
                    str(datesdict['dates']['session']),
                    str(datesdict['dates']['labview']),
                    str(datesdict['dates']['bloodgas']),
                    str(datesdict['dates']['konica']),
                    str(datesdict['dates']['pulseox']) if datesdict['dates']['pulseox'] != [] else 'No manual pulse ox entry data',
                ],
                'Time': [
                    "",
                    labview_timestamp.time() if abg_timestamp is not None and labview_timestamp is not None else 'No sample',
                    abg_timestamp.time() if abg_timestamp is not None and labview_timestamp is not None else 'No sample',
                    pd.to_datetime(automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'konica_timestamp'].values[0]).time() if pd.notnull(automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'konica_timestamp'].values[0]) else "",
                    "",
                ]
            }

            dates_df = pd.DataFrame(dates_data).set_index('Label')

            # Display the DataFrame as a table
            st.table(dates_df)
            # st.markdown(f'''
            
            # * **Session date:** {datesdict['dates']['session']}

            # * **Labview date:** {datesdict['dates']['labview']}

            # * **Konica date:** {datesdict['dates']['konica']}

            # * **ABG file date:** {datesdict['dates']['bloodgas']}

            # * **Manual pulse ox entry date:** {datesdict['dates']['pulseox'] if datesdict['dates']['pulseox'] != [] else 'No manual pulse ox entry data'}

            # * Sample {abg_timestamp_num} (Labview): {frame.loc[frame['sample'] == abg_timestamp_num, 'Timestamp'].values[0] if abg_timestamp_num in frame['sample'].values else 'No sample'} 

            # * Sample {abg_timestamp_num} (ABG): {re.search(r'Sample \d*:(.*)', abg_timestamp).group(1) if abg_timestamp_num in frame['sample'].values else 'No sample 2'}

            # ''')
        with right:
            st.markdown('### Patient ID Discrepancies')
            st.checkbox('Patient ID Discrepancies addressed', key='qc_id_discrepancy')

            # make a warning flag if there is patient_id_issues_tf = True
            if automated_qc_df.loc[automated_qc_df['session_id']==selected_session,'patient_id_issues_tf'].values[0]:
                st.error('Patient ID discrepancies found', icon="🚨")
            else:
                st.success('No patient ID discrepancies found', icon="✅")
            
            # Create a DataFrame with the patient IDs
            patient_ids = {
                'File': ['Session patient ID', 'Konica patient ID', 'ABG file patient ID', 'Manual pulse ox entry patient ID'],
                'Patient ID': [
                    str(ptids['patient_ids']['session']),
                    str(ptids['patient_ids']['konica']),
                    str(ptids['patient_ids']['bloodgas']),
                    str(ptids['patient_ids']['pulseox'])
                ]
            }

            # Convert the dictionary to a DataFrame
            

            # Display the DataFrame as a table
            st.table(pd.DataFrame(patient_ids).set_index('File'))
            # st.markdown(f'''
            # * **Session patient ID**: {ptids['patient_ids']['session']} 

            # * **Konica patient ID**: {ptids['patient_ids']['konica']}

            # * **ABG file patient ID**: {ptids['patient_ids']['bloodgas']}

            # * **Manual pulse ox entry patient ID**: {ptids['patient_ids']['pulseox']}
            # ''')
        
        st.markdown('### Data Quality Check')
        st.checkbox('Data quality checked (no bad points)', key='qc_quality')
        st.checkbox('Needs further review or action', key='qc_data_qualtiy_action')
        one, two = st.columns(2)
        with one:
            st.write(f'''
         * Crosses indicate that the data point was rejected by the algorithm (either so2 or Nellcor). 
         * Nellcor: Red outlines indicate Nellcor values where the bias is > {max_bias}, but were not cleaned out.
        * Bias: Blue outlines indicate bias values > {max_bias}.
         ''')
            
        with two:
            st.write('**Max bias:**', round(frame['bias'].abs().max(),2))
            st.write('**ARMS calculations:**')
            armsdf = {'Device': [key for key, val in automated_qc_df.loc[automated_qc_df['session_id'] == selected_session, 'arms'].values[0].items() ],
                      'ARMS (clean)': [f"{val:.2f}" for key, val in automated_qc_df.loc[automated_qc_df['session_id'] == selected_session, 'arms'].values[0].items() ],
                      # this handles the fact that the ARMS device names can be Masimo 97 or Rad97-60, but in the frame it is always Masimo 97
                      'ARMS (all)': [f"{ox.arms(labview_samples[labview_samples['session'] == selected_session][device], labview_samples[labview_samples['session'] == selected_session]['so2']):.2f}" if device in ['Masimo 97/SpO2', 'Nellcor/SpO2', 'Nellcor PM1000N-1/SpO2'] else f"{ox.arms(frame['Masimo 97/SpO2'], frame['so2']):.2f}" for device in automated_qc_df.loc[automated_qc_df['session_id'] == selected_session, 'arms'].values[0].keys()]
                    }
            
            st.table(pd.DataFrame(armsdf).set_index('Device'))

            # # temp comment out until proven stable
            # if not pd.isna(automated_qc_df.loc[automated_qc_df['session_id'] == selected_session, 'arms'].values[0]):    
            #     for value in automated_qc_df.loc[automated_qc_df['session_id'] == selected_session, 'arms'].values[0].items():
            #         st.write(value[0], round(value[1],2))

                        
            # ############# Danni Updates ###############
            # st.write('**ARMS comparison:**')
            # if labview_samples[labview_samples['session'] == selected_session]['Nellcor PM1000N-1/SpO2'].sum() > 0:
            #     all_data_arms = ox.arms(labview_samples[labview_samples['session'] == selected_session]['Rad97-60/SpO2'], labview_samples[labview_samples['session'] == selected_session]['so2'])
            #     st.write('Masimo ARMS (all data):', round(all_data_arms,2))
            #     cleaned_data_arms = ox.arms(labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['Rad97-60/SpO2'], labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['so2'])
            #     st.write('Masimo ARMS (cleaned data):', round(cleaned_data_arms,2))
            
            #     all_data_arms = ox.arms(labview_samples[labview_samples['session'] == selected_session]['Nellcor PM1000N-1/SpO2'], labview_samples[labview_samples['session'] == selected_session]['so2'])
            #     st.write('Nellcor ARMS (all data):', round(all_data_arms,2))
            #     cleaned_data_arms = ox.arms(labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['Nellcor PM1000N-1/SpO2'], labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['so2'])
            #     st.write('Nellcor ARMS (cleaned data):', round(cleaned_data_arms,2))
            # else:
            #     all_data_arms = ox.arms(labview_samples[labview_samples['session'] == selected_session]['Masimo 97/SpO2'], labview_samples[labview_samples['session'] == selected_session]['so2'])
            #     st.write('Masimo ARMS (all data):', round(all_data_arms,2))
            #     cleaned_data_arms = ox.arms(labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['Masimo 97/SpO2'], labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['so2'])
            #     st.write('Masimo ARMS (cleaned data):', round(cleaned_data_arms,2))
            
            #     all_data_arms = ox.arms(labview_samples[labview_samples['session'] == selected_session]['Nellcor/SpO2'], labview_samples[labview_samples['session'] == selected_session]['so2'])
            #     st.write('Nellcor ARMS (all data):', round(all_data_arms,2))
            #     cleaned_data_arms = ox.arms(labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['Nellcor/SpO2'], labview_samples[(labview_samples['session'] == selected_session) & labview_samples['algo_status']]['so2'])
            #     st.write('Nellcor ARMS (cleaned data):', round(cleaned_data_arms,2))
            # ##########################################

            st.text_area('Data quality notes', key='qc_quality_notes')

        if frame is not None:
            plotcolumns = ['so2', 'Nellcor/SpO2', 'Masimo 97/SpO2', 'bias']
            st.plotly_chart(create_plot(frame, plotcolumns, limit_to_manual_sessions), use_container_width=True)
            frame['Nellcor/PI'] = frame['Nellcor/PI'].apply(lambda x: x/10 if pd.notnull(x) else x)
            st.dataframe(frame.set_index('sample')
                        .drop(columns=[ 'RR','Masimo HB/SpO2', 'Masimo HB/PI', 'Masimo HB/SpO2_diff_prev', 'Masimo HB/SpO2_diff_next',### Danni doesn't think there is Masimo HB/SpO2?
                                        'Masimo 97/HR', 'Nellcor/HR', 'Masimo HB/HR', 'Rad97-60/HR', 'Nellcor PM1000N-1/HR', 'RR', ### They were added by Danni for labview_samples but don't need to be shown on QC dashboard
                                        'sample_diff_prev',
                                        'sample_diff_next',
                                        # 'Nellcor/SpO2_diff_prev',
                                        # 'Nellcor/SpO2_diff_next',
                                        'Masimo 97/SpO2_diff_prev',
                                        'Masimo 97/SpO2_diff_next',
                                        'so2_symbol',
                                        'Nellcor/SpO2_symbol',
                                        'bias_symbol',
                                        'so2_line',
                                        'Nellcor/SpO2_line',
                                        'Masimo 97/SpO2_line',
                                        'Masimo 97/SpO2_symbol',
                                        'bias_line'])
                        [['Masimo 97/SpO2', 'Masimo 97/PI', 'Nellcor/SpO2', 'Nellcor/PI', 'so2', 'bias', 'Nellcor/SpO2_diff_prev', 'Nellcor/SpO2_diff_next', 'so2_diff_prev', 'so2_diff_next', 'Timestamp', 'Timestamp_diff_prev', 'Timestamp_diff_next', 'so2_stable', 'so2_reason', 'Nellcor_stable', 'Nellcor_reason', 'Masimo_stable', 'Masimo_reason', 'algo_status', 'algo', 'manual_so2', 'manual_algo_compare']],
                        use_container_width=True)
        
        st.markdown('### Final QC')
        st.checkbox('QC complete', key='qc_complete')
        qc_message = st.container()
        
        submit_button = st.form_submit_button('Save QC', on_click=update_qc_field, args=(qc_status, selected_session))
else:
    st.write('No session selected')