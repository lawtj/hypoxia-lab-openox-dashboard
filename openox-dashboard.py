import streamlit as st
import pandas as pd
import numpy as np
import os
from redcap import Project
import matplotlib.pyplot as plt
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
        #run the jupyter notebook
        from nbtopy import *
        for i,j in zip([db, haskonica, hasmonk, hasboth, haskonica_notmonk, hasmonk_notkonica, db_style, column_dict],['db', 'haskonica', 'hasmonk', 'hasboth', 'haskonica_notmonk', 'hasmonk_notkonica','db_style','column_dict']):
            st.session_state[j] = i
    st.write('loaded from redcap')
else:
    db = st.session_state['db']
    db_style = st.session_state['db_style']
    haskonica = st.session_state['haskonica']
    hasmonk = st.session_state['hasmonk']
    hasboth = st.session_state['hasboth']
    haskonica_notmonk = st.session_state['haskonica_notmonk']
    hasmonk_notkonica = st.session_state['hasmonk_notkonica']
    column_dict = st.session_state['column_dict']
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

# style database
def highlight_value_greater(s, cols_to_sum,threshold):
    sums = db[cols_to_sum].sum(axis=1)
    mask = s > sums*threshold
    return ['background-color: #b5e7a0' if v else '' for v in mask]

db.rename(columns=column_dict, inplace=True)


st.write('now using Monk Dorsal')

st.subheader('Feature requests')
st.markdown('''
* Filters for...?
* Device names
            ''')

st.dataframe(db
        .style
        .apply(highlight_value_greater,cols_to_sum=['Monk ABC','Monk DEF','Monk HIJ'], threshold=.25, subset=['Monk ABC'])
        .apply(highlight_value_greater, cols_to_sum=['Monk ABC','Monk DEF','Monk HIJ'], threshold=.25, subset=['Monk DEF'])
        .apply(highlight_value_greater, cols_to_sum=['Monk ABC','Monk DEF','Monk HIJ'], threshold=.25, subset=['Monk HIJ'])
        #now style column ita>25 with threshold of .25
        .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA >25'])
        # same with ita25to-35
        .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA 25 to -35'])
        # and same for ita<-35
        .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA <-35'])
        #highlight if number of patients with ITA<-45 >= 2 
        .map(lambda x: 'background-color: #b5e7a0' if x>=2 else "", subset=['ITA <-45'])
        .format(lambda x: f'{x:,.0f}', subset=list(column_dict.values()))
        .background_gradient(subset=['Any ITA'], cmap='RdYlGn')
             ,column_config={
                "Unique Patients": st.column_config.ProgressColumn(
                    "Unique Patients",
                    help="Number of unique patients",
                    format="%f",
                    min_value=0,
                    max_value=int(db['Unique Patients'].max())),
                "ITA <-45": st.column_config.ProgressColumn(
                    "ITA <-45",
                    help="Number of patients with ITA<-45",
                    format="%f",
                    min_value=0,
                    max_value=2)
    },use_container_width=True)

# st.subheader('Data problems')
# st.markdown(db
#         .style
#         .apply(highlight_value_greater,cols_to_sum=['Monk ABC','Monk DEF','Monk HIJ'], threshold=.25, subset=['Monk ABC'])
#         .apply(highlight_value_greater, cols_to_sum=['Monk ABC','Monk DEF','Monk HIJ'], threshold=.25, subset=['Monk DEF'])
#         .apply(highlight_value_greater, cols_to_sum=['Monk ABC','Monk DEF','Monk HIJ'], threshold=.25, subset=['Monk HIJ'])
#         #now style column ita>25 with threshold of .25
#         .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA >25'])
#         # same with ita25to-35
#         .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA 25 to -35'])
#         # and same for ita<-35
#         .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA <-35'])
#         #highlight if number of patients with ITA<-45 >= 2 
#         .map(lambda x: 'background-color: #82b74b' if x>=2 else "", subset=['ITA <-45'])
#         .format(lambda x: f'{x:,.0f}', subset=list(column_dict.values()))
#         .background_gradient(subset=['Any ITA'], cmap='RdYlGn')
#         .to_html(), unsafe_allow_html=True)