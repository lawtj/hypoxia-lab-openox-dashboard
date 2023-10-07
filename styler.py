import streamlit as st
import pandas as pd
import numpy as np


# example multi-index dataframe
data = {'level_0': [1, 2, 3],
        'level_1': ['A', 'B', 'C'],
        '01/23/2023': ['20%', '30%', '40%'],
        '01/24/2023': ['10%', '20%', '30%'],
        '01/25/2023': ['15%', '25%', '35%'],
        '01/26/2023': ['5%', '10%', '15%'],
        '01/27/2023': ['1%', '2%', '3%']}

df = pd.DataFrame(data)
df = df.set_index(['level_0', 'level_1'])

def make_bar_style(x):
    if '%' in str(x):
        x = float(x.strip('%'))
        return f"background: linear-gradient(90deg,#5fba7d {x}%, transparent {x}%); width: 10em"  
    return ''


st.dataframe(df.style.applymap(make_bar_style))