import streamlit as st
import pandas as pd
import numpy as np
import os
from redcap import Project
st.set_page_config(layout="wide")

st.title('OpenOx Dashboard')

from hypoxialab_functions import *

if 'db' not in st.session_state:
    session = st_load_project('REDCAP_SESSION')
    session = session.reset_index()
    manual = st_load_project('REDCAP_MANUAL') 
    participant = st_load_project('REDCAP_PARTICIPANT')
    konica = st_load_project('REDCAP_KONICA')
    manual = reshape_manual(manual)
    is_streamlit = True
    with st.spinner('Loading data from Redcap...'):
        from nbtopy import *
        for i,j in zip([db, haskonica, hasmonk, hasboth, haskonica_notmonk, hasmonk_notkonica, db_style],['db', 'haskonica', 'hasmonk', 'hasboth', 'haskonica_notmonk', 'hasmonk_notkonica','db_style']):
            st.session_state[j] = i
    st.write('loaded from redcap')
else:
    db = st.session_state['db']
    db_style = st.session_state['db_style'rr]
    haskonica = st.session_state['haskonica']
    hasmonk = st.session_state['hasmonk']
    hasboth = st.session_state['hasboth']
    haskonica_notmonk = st.session_state['haskonica_notmonk']
    hasmonk_notkonica = st.session_state['hasmonk_notkonica']
    st.write('loaded from session state')

###### layout ######
st.subheader('Count of patients')

one,two,three = st.columns(3)

one.metric('Number of patients with Konica data', len(haskonica))
two.metric('Number of patients with Monk data', len(hasmonk))
three.metric('Number of patients with both Konica and Monk data', len(hasboth))

one, two = st.columns(2)
one.metric('Has Konica but not Monk', len(haskonica_notmonk))
two.metric('Has Monk but not Konica', len(hasmonk_notkonica))

#st.dataframe(db, use_container_width=True)

st.dataframe(db_style, use_container_width=True)

st.subheader('Data problems')
st.write('we are missing birthdays for some people, so we cannot calculate age for them')
st.write('ID# 1132 has an incorrect birthday, listed as this year, which is causing minimum age to be listed as zero')
st.write('something wrong with BMI calculation for ID# 1144, entered as 10.4')