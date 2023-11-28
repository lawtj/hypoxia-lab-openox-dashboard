import streamlit as st
import pandas as pd
import numpy as np
import os
from redcap import Project
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

from hypoxialab_functions import *

st.title('OpenOx Dashboard')

if 'db' not in st.session_state:
    print('db is not in session state')
    with st.spinner('Loading data from Redcap...'):
        #run the jupyter notebook
        from nbtopy import *
        for i,j in zip([db, haskonica, hasmonk, hasboth, haskonica_notmonk, hasmonk_notkonica, column_dict],['db', 'haskonica', 'hasmonk', 'hasboth', 'haskonica_notmonk', 'hasmonk_notkonica','column_dict']):
            st.session_state[j] = i
    st.write('loaded from redcap')
else:
    db = st.session_state['db']
    haskonica = st.session_state['haskonica']
    hasmonk = st.session_state['hasmonk']
    hasboth = st.session_state['hasboth']
    haskonica_notmonk = st.session_state['haskonica_notmonk']
    hasmonk_notkonica = st.session_state['hasmonk_notkonica']
    column_dict = st.session_state['column_dict']
    st.write('loaded from session state')


###### layout ######
st.subheader('Count of subjects')

one,two,three = st.columns(3)

one.metric('Number of subjects with Konica data', len(haskonica))
two.metric('Number of subjects with Monk data', len(hasmonk))
three.metric('Number of subjects with both Konica and Monk data', len(hasboth))

one, two = st.columns(2)
one.metric('Has Konica but not Monk', len(haskonica_notmonk))
two.metric('Has Monk but not Konica', len(hasmonk_notkonica))

#st.dataframe(db, use_container_width=True)

# style database
db.rename(columns=column_dict, inplace=True)

def highlight_value_greater(s, cols_to_sum, threshold):
    sums = db[cols_to_sum].sum(axis=1)
    mask = (s >= sums*threshold) & (sums > 0)
    return ['background-color: #b5e7a0' if v else '' for v in mask]

st.write('Using Monk Dorsal, and ITA dorsal (median)')

st.subheader('Feature requests')
st.markdown('''
* Filters for...?
* Device names
            ''')

st.dataframe(db
        .style
        # Highlight if Sample size >= 24 (unique patient_id)
        .map(lambda x: 'background-color: #b5e7a0' if x>=24 else "", subset=['Unique Subjects'])

         # Highlight if Average of number of data points per participant/session = 24 (+/-4) (sao2)
        .map(lambda x: 'background-color: #b5e7a0' if x>=20 and x<=28 else "", subset=['Avg Samples per Session'])

        # Highlight if Range of number of data points per participant/session = 17-30 (sao2)
        .map(lambda x: 'background-color: #b5e7a0' if x==1 else "", subset=['17 <= Num Samples per Session <= 30'])

        # Highlight if Each decade between the 70% - 100% saturations contains 33% of the data points (sao2)
        .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 70-80'])
        .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 80-90'])
        .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 90-100'])

         # Highlight if >= 90% of the sessions in the same device provide so2 data < 85 (sao2)
        .map(lambda x: 'background-color: #b5e7a0' if x>=90 else "", subset=['%\n of Sessions Provides SaO2 < 85'])

        # Highlight if >=70% participants/sessions provide data points in the 70%-80% decade (sao2)
        .map(lambda x: 'background-color: #b5e7a0' if x>=70 else "", subset=['%\n of Sessions Provides SaO2 in 70-80'])

        # Highlight if Each sex has approximately 40% percentage (assigned_sex)
        .apply(highlight_value_greater, cols_to_sum=['Female','Male'], threshold=.40, subset=['Female'])
        .apply(highlight_value_greater, cols_to_sum=['Female','Male'], threshold=.40, subset=['Male'])

        # Highlight if >= 1 in each of the 10 MST categories (monk_dorsal)
        .map(lambda x: 'background-color: #b5e7a0' if x==10 else "", subset=['Unique Monk'])

        # Highlight if >= 25% in each of the following MST categories: 1-4, 5-7, 8-10
        .apply(highlight_value_greater,cols_to_sum=['Monk ABCD','Monk EFG','Monk HIJ'], threshold=.25, subset=['Monk ABCD'])
        .apply(highlight_value_greater, cols_to_sum=['Monk ABCD','Monk EFG','Monk HIJ'], threshold=.25, subset=['Monk EFG'])
        .apply(highlight_value_greater, cols_to_sum=['Monk ABCD','Monk EFG','Monk HIJ'], threshold=.25, subset=['Monk HIJ'])
        
        # Highlight if >= 25% in each of the following MST categories: 1-4(>25°), 5-7(>-35°, <=25°), 8-10(<=-35°)
        .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA > 25'])
        .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['-35 < ITA <= 25'])
        .apply(highlight_value_greater, cols_to_sum=['Any ITA'], threshold=.25, subset=['ITA <= -35'])

        # Highlight if >=1 subject in category MST 1-4 with ITA >= 50°
        .map(lambda x: 'background-color: #b5e7a0' if x>=1 else "", subset=['ITA >= 50 & Monk ABCD'])

        # Highlight if >=2 subjects in category MST 8-10 with ITA <= -45° 
        .map(lambda x: 'background-color: #b5e7a0' if x>=2 else "", subset=['ITA <= -45 & Monk HIJ'])

        # .format(lambda x: f'{x:,.2f}', subset=list(column_dict.values())),
        .format(lambda x: f'{x:,.0f}', subset=list(column_dict.values())),

        height = (23 + 1) * 35 + 3)