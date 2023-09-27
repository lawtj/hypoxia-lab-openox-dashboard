import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'cost': [25.99, 97.45, 64.32, 14.78],
    'grams': [101.89, 20.924, 50.12, 40.015]
    })

st.dataframe(df)

st.dataframe(df.style.highlight_max(axis=0))

st.dataframe(df, column_config={'cost': st.column_config.NumberColumn('Cost',format="$%.2g"),
                            'grams': st.column_config.NumberColumn('Grams',format="%.2f")})

st.dataframe(df.style.highlight_max(axis=0), 
             column_config={'cost': st.column_config.NumberColumn('Cost',format="$%.2g"),
                            'grams': st.column_config.NumberColumn('Grams',format="%.2f")})

st.dataframe(
    df.style
)