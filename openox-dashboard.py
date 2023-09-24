import streamlit as st
import pandas as pd
import numpy as np
import os
from redcap import Project
st.set_page_config(layout="wide")

st.title('OpenOx Dashboard')

from hypoxialab_functions import *

with st.spinner('Loading data from Redcap...'):
    from nbtopy import *

# set wide by default

###### layout ######
st.subheader('Count of patients')

one,two,three = st.columns(3)

one.metric('Number of patients with Konica data', len(haskonica))
two.metric('Number of patients with Monk data', len(hasmonk))
three.metric('Number of patients with both Konica and Monk data', len(hasboth))

one, two = st.columns(2)
one.metric('Has Konica but not Monk', len(haskonica_notmonk))
two.metric('Has Monk but not Konica', len(hasmonk_notkonica))

st.write(db)

st.write('we are missing birthdays for some people, so we cannot calculate age for them')