import streamlit as st
import pandas as pd
import hypoxialab_functions as ox

# pull data from redcap
participant = ox.st_load_project('REDCAP_PARTICIPANT')
participant = participant.reset_index()
participant = participant.drop(columns=['subject_id'])
participant = participant.rename(columns={'record_id':'subject_id'})
participant = participant[participant['dob'].between('1960-01-01','2007-12-31')]
session = ox.st_load_project('REDCAP_SESSION')
session = session.reset_index()
session = session.rename(columns={'record_id':'session','patient_id':'subject_id'})

# fields Ella wants from participant stay consistent: height, weight, BMI, sex, race, ethnicity, dob
participant = participant[['subject_id', 'height', 'weight', 'bmi', 'assigned_sex', 'race', 'ethnicity', 'dob']]
# merge participant and session data
participant_session = pd.merge(participant, session, on='subject_id', how='right')
# calculate age
participant_session['session_date'] = pd.to_datetime(participant_session['session_date'])
participant_session['dob'] = pd.to_datetime(participant_session['dob'])
participant_session['age'] = round((participant_session['session_date'] - participant_session['dob']).dt.days / 365.25, 0)
# edit values in some of the fields
racedict = {
    '1': 'African American',
    '2': 'African American Caucasian',
    '3': 'African American Other/Mutiethnic Caucasian',
    '4': 'Asian',
    '5': 'Asian African American Caucasian Other/Mutiethnic',
    '6': 'Asian Caucasian',
    '7': 'Asian Caucasian Other/Mutiethnic',
    '8': 'Asian Hawaiin/Pacific Islander',
    '9': 'Asian Other/Mutiethnic',
    '10': 'Caucasian',
    '11': 'Caucasian African American Hispanic',
    '12': 'Caucasian Asian',
    '13': 'Caucasian Hispanic',
    '14': 'Caucasian Native American/Alaskan',
    '15': 'Caucasian Other/Mutiethnic',
    '16': 'Hawaiin/Pacific Islander',
    '17': 'Hispanic',
    '18': 'Hispanic Caucasian',
    '19': 'Native American/Alaskan',
    '20': 'Other/Mutiethnic',
    '21': 'Other/Mutiethnic African American',
    '22': 'Other/Mutiethnic Asian',
    '23': 'Other/Mutiethnic Asian Caucasian',
    '24': 'Other/Mutiethnic Caucasian',
    '25': 'Other/Mutiethnic Caucasian Asian',
    '26': 'Other/Mutiethnic Hispanic',
    # now the custom edits
    '2, Hispanic': 'Hispanic',
    'Asian\nOther/Mutiethnic': 'Other/Mutiethnic Asian',
    'Caucasian Asian': 'Asian Caucasian',
}
participant_session.race.replace(racedict, inplace=True)

### Demographic Data Table
st.title('Demographic Data Table')

selected_session = st.multiselect('Select session IDs for the study', participant_session['session'].sort_values(ascending=False))
selected_fields = st.multiselect('Select fields to be included in the table', session.columns)

# always include height, weight, bmi, assigned_sex, race, ethnicity, age in the table
ordered_fields = selected_fields + ['age', 'height', 'weight', 'bmi', 'assigned_sex', 'race', 'ethnicity']
# Create the output table with the correct order of fields
output_table = participant_session.loc[participant_session['session'].isin(selected_session), ordered_fields]
st.dataframe(output_table, use_container_width=True)

### Subject Pool Planning
st.title('Subject Pool Planning')

planned_subjects = st.multiselect('Select subject IDs for the study', participant_session['subject_id'].sort_values(ascending=False))

planned_subjects_df = participant_session[participant_session['subject_id'].isin(planned_subjects)].drop_duplicates(subset=['subject_id'], keep='last') # keeping Monk data from the last session
st.dataframe(planned_subjects_df[['subject_id', 'assigned_sex', 'monk_forehead', 'session_notes']].sort_values(by='subject_id'), use_container_width=True)
st.dataframe(planned_subjects_df['assigned_sex'].value_counts(normalize=True).reset_index())
st.dataframe(planned_subjects_df.sort_values(by='monk_forehead')['monk_forehead'].value_counts().reset_index())