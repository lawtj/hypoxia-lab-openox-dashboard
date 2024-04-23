import streamlit as st
import pandas as pd
import numpy as np
from redcap import Project
import io
import plotly.express as px

def check_password():
    """Returns `True` if the user had one of the correct passwords."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["device_performance_pw"]:
            st.session_state["password_correct"] = True
            st.session_state['internal_team'] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    st.set_page_config(layout="wide")

    if 'merged_cleaned' not in st.session_state:
        api_url = 'https://redcap.ucsf.edu/api/'
        api_k = st.secrets['REDCAP_DANNI']
        proj = Project(api_url, api_k)
        f = io.BytesIO(proj.export_file(record='1', field='file')[0])
        merged_cleaned = pd.read_parquet(f)
        st.session_state['merged_cleaned'] = merged_cleaned
    else:
        st.caption('Using cached data')
        merged_cleaned = st.session_state['merged_cleaned']
        
    st.header('Device Performance')
    st.subheader('Select a device to view its performance')

    device_manufacturer = merged_cleaned[['device', 'manufacturer']].drop_duplicates().reset_index(drop=True)
    device_manufacturer['device_manufacturer'] = list(zip(device_manufacturer['device'], device_manufacturer['manufacturer']))
    device_manufacturer = device_manufacturer.drop(columns=['device', 'manufacturer'])
    device_selected = st.selectbox('', device_manufacturer)
    df_selected = merged_cleaned[merged_cleaned['device'] == device_selected[0]]
    df_selected.rename(columns={'patient_id':'subject'}, inplace=True)
    df_selected['absolute bias'] = abs(df_selected['saturation'] - df_selected['so2'])

    one, two = st.columns(2)
    with one:
        st.write("Manufacturer: ", df_selected['manufacturer'].unique()[0])
        st.write("Model: ", df_selected['model'].unique()[0])
        st.write("Number of Subjects: " + str(df_selected['subject'].nunique()))
        st.write("Number of Sessions: " + str(df_selected['session'].nunique()))
        def arms(spo2,sao2):
            return np.sqrt(np.mean((spo2-sao2)**2))
        st.write("ARMS: " + str(arms(df_selected['saturation'],df_selected['so2']).round(2)))
        # st.write("Differential bias between the extreme ITAs")
        # NDB_results_ITA = calc_NDB(df_selected)
        # st.write(NDB_results_ITA)
        
    with two:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['Mean Abs. Bias per Subject', 'ITA Distribution', 'Monk Distribution', 'Mean Abs. Bias vs. ITA','Mean Abs. Bias vs. Monk'])
        with tab1:
            mean_bias_per_subject = df_selected.groupby('subject')['absolute bias'].mean().reset_index().sort_values(by='absolute bias')
            mean_bias_per_subject.rename(columns={'absolute bias':'mean absolute bias'}, inplace=True)
            fig = px.scatter(mean_bias_per_subject, x='subject', y='mean absolute bias', title='Mean Absolute Bias per Subject').update_xaxes(showticklabels=False, ).to_dict()
            st.plotly_chart(fig)
        with tab2:
            df = df_selected.groupby('encounter_id')['ita'].mean().reset_index().sort_values(by='ita')
            fig = px.scatter(df, x='encounter_id', y='ita', title='ITA Measurements at Dorsal per Subject').update_xaxes(showticklabels=False, )
            st.plotly_chart(fig)
        with tab3:
            df = df_selected.groupby('encounter_id')['monk_forehead'].unique().sort_values().reset_index()
            df['monk_forehead'] = df['monk_forehead'].apply(lambda x: [chr(64+int(i)) for i in x])
            df = df.explode('monk_forehead')
            st.plotly_chart(px.histogram(df, x=df['monk_forehead'], title='MST at Forehead per Subject', text_auto=True))
        with tab4:
            ita = df_selected.groupby('encounter_id')['ita'].mean().reset_index()
            bias = df_selected.groupby('encounter_id')['absolute bias'].mean().reset_index()
            df = pd.merge(ita, bias, on='encounter_id')
            df.rename(columns={'ita':'ITA', 'absolute bias':'Mean Absolute Bias'},inplace=True)
            fig = px.scatter(df, x='ITA', y='Mean Absolute Bias', title='Mean Absolute Bias vs. ITA Dorsal (per Subject)').to_dict()
            st.plotly_chart(fig)
        with tab5:
            monk = df_selected.groupby('encounter_id')['monk_forehead'].unique().sort_values().reset_index()
            monk['monk_forehead'] = monk['monk_forehead'].apply(lambda x: [chr(64+int(i)) for i in x])
            monk = monk.explode('monk_forehead')
            bias = df_selected.groupby('encounter_id')['absolute bias'].mean().reset_index()
            df = pd.merge(monk, bias, on='encounter_id')
            df.rename(columns={'monk_forehead':'MST', 'absolute bias':'Mean Absolute Bias'}, inplace=True)
            fig = px.scatter(df, x='MST', y='Mean Absolute Bias', title='Mean Absolute Bias vs. MST Forehead (per Subject)').to_dict()
            st.plotly_chart(fig)
        

