import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")
import create_figure

st.title('OpenOx Dashboard')

if not ('db_new_v1' in st.session_state or 'db_new_v2' in st.session_state or 'db_old' in st.session_state):
    print('db is not in session state')
    with st.spinner('Loading data from Redcap...'):
        from nbtopy import db_new_v1,db_new_v2, db_old, haskonica, hasmonk, hasboth, haskonica_notmonk, hasmonk_notkonica, column_dict_db_new_v1, column_dict_db_new_v2, column_dict_db_old, konica, session, joined
        for i,j in zip([db_new_v1, db_new_v2, db_old, haskonica, hasmonk, hasboth, haskonica_notmonk, hasmonk_notkonica, column_dict_db_new_v1, column_dict_db_new_v2, column_dict_db_old, konica, session, joined],['db_new_v1', 'db_new_v2', 'db_old', 'haskonica', 'hasmonk', 'hasboth', 'haskonica_notmonk', 'hasmonk_notkonica','column_dict_db_new_v1', 'column_dict_db_new_v2', 'column_dict_db_old', 'konica','session', 'joined']):
            st.session_state[j] = i
    st.write('loaded from redcap')
else:
    print('loading from session state')
    db_new_v1 = st.session_state['db_new_v1']
    db_new_v2 = st.session_state['db_new_v2']
    db_old = st.session_state['db_old']
    haskonica = st.session_state['haskonica']
    hasmonk = st.session_state['hasmonk']
    hasboth = st.session_state['hasboth']
    haskonica_notmonk = st.session_state['haskonica_notmonk']
    hasmonk_notkonica = st.session_state['hasmonk_notkonica']
    column_dict_db_new_v1 = st.session_state['column_dict_db_new_v1']
    column_dict_db_new_v2 = st.session_state['column_dict_db_new_v2']
    column_dict_db_old = st.session_state['column_dict_db_old']
    konica = st.session_state['konica']
    session = st.session_state['session']
    joined = st.session_state['joined']
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

st.write('Using Monk forehead. Using ITA dorsal (median) if device is fingertip.')
st.write('SaO2 in 70-80 range -> 67-80')

def highlight_value_greater(s, cols_to_sum, threshold):
    sums = db_new_v1[cols_to_sum].sum(axis=1)
    mask = (s >= sums*threshold) & (sums > 0)
    return ['background-color: #b5e7a0' if v else '' for v in mask]

st.subheader('Choose a version of the standard to look at:')

selected_df = st.radio('', ("ISO 2017/ FDA 2013", "ISO 2023/FDA 2024", "ISO 2024/FDA 2024"), index=2)

if selected_df == "ISO 2017/ FDA 2013":
    db_old.rename(columns=column_dict_db_old, inplace=True)
    st.dataframe(db_old
                 .style
                 # Highlight if Sample size >= 10 (unique sessions)
                 .map(lambda x: 'background-color: #b5e7a0' if x>=10 else "", subset=['Unique Sessions'])
                
                 # Highlight if Average of number of data points per participant/session = 25 (sao2) 
                 .map(lambda x: 'background-color: #b5e7a0' if x>24 and x<26 else "", subset=['Avg Samples per Session'])
                
                 # Highlight if minimum 20 data points per subject/session (sao2)
                 .map(lambda x: 'background-color: #b5e7a0' if x>=10 else "", subset=['Unique Sessions with >= 20 Samples'])
                
                 # Highlight if Each decade between the 70% - 100% saturations contains 33% of the data points (sao2)
                 .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 70-80 (pooled)'])
                 .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 80-90 (pooled)'])
                 .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 90-100 (pooled)'])
                
                  # Highlight if for 70-100% include 73% and 97% (sao2)
                 .map(lambda x: 'background-color: #b5e7a0' if x<=73 and x!=0 else "", subset=['Min SaO2'])
                 .map(lambda x: 'background-color: #b5e7a0' if x>=97 else "", subset=['Max SaO2'])
                
                 # Highlight if >= 2 with dark skin
                 .map(lambda x: 'background-color: #b5e7a0' if x>=2 else "", subset=['# Sessions with Fitzpatrick V or VI'])
                
                 # .format(lambda x: f'{x:,.2f}', subset=list(column_dict.values())),
                 .format(lambda x: f'{x:,.0f}', subset=list(column_dict_db_old.values())),

                 height = (23 + 1) * 35 + 3)

if selected_df == "ISO 2023/FDA 2024":
    # style database
    db_new_v1.rename(columns=column_dict_db_new_v1, inplace=True)
    st.dataframe(db_new_v1
            .style
            # Highlight if Sample size >= 24 (unique patient_id)
            .map(lambda x: 'background-color: #b5e7a0' if x>=24 else "", subset=['Unique Subjects'])

            # Highlight if Average of number of data points per participant/session = 24 (+/-4) (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=20 and x<=28 else "", subset=['Avg Samples per Session'])

            # Highlight if Range of number of data points per participant is in 17-30 (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=24 else "", subset=['Unique Subjects with 17-30 Samples'])

            # Highlight if Each decade between the 70% - 100% saturations contains 33% of the data points (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 70-80 (pooled)'])
            .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 80-90 (pooled)'])
            .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 90-100 (pooled)'])
            
             # Highlight if for 70-100% include 73% and 97% (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x<=73 else "", subset=['Min SaO2'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=97 else "", subset=['Max SaO2'])

            # Highlight if >= 90% of the sessions in the same device provide so2 data < 85 (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=90 else "", subset=['%\n of Sessions Provides SaO2 < 85'])

            # Highlight if >=70% participants/sessions provide data points in the 70%-80% decade (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=70 else "", subset=['%\n of Sessions Provides SaO2 in 70-80'])

            # Highlight if Each sex has approximately 40% percentage (assigned_sex)
            # .apply(highlight_value_greater, cols_to_sum=['Female','Male'], threshold=.40, subset=['Female'])
            # .apply(highlight_value_greater, cols_to_sum=['Female','Male'], threshold=.40, subset=['Male'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=9.6 else "", subset=['Female'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=9.6 else "", subset=['Male'])

            # Highlight if >= 1 in each of the 10 MST categories (monk_forehead)
            .map(lambda x: 'background-color: #b5e7a0' if x==10 else "", subset=['Unique Monk Forehead'])

            # Highlight if >= 25% in each of the following MST categories: 1-4, 5-7, 8-10 (monk_forehead)
            # .apply(highlight_value_greater,cols_to_sum=['Monk ABCD','Monk EFG','Monk HIJ'], threshold=.25, subset=['Monk ABCD'])
            # .apply(highlight_value_greater, cols_to_sum=['Monk ABCD','Monk EFG','Monk HIJ'], threshold=.25, subset=['Monk EFG'])
            # .apply(highlight_value_greater, cols_to_sum=['Monk ABCD','Monk EFG','Monk HIJ'], threshold=.25, subset=['Monk HIJ'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['Monk ABCD'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['Monk EFG'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['Monk HIJ'])
            
            # Highlight if >= 25% in each of the following MST categories: 1-4(>25°), 5-7(>-35°, <=25°), 8-10(<=-35°) (monk_forehead & ita_dorsal)
            # .apply(highlight_value_greater, cols_to_sum=['Unique Subjects'], threshold=.25, subset=['ITA > 25 & Monk ABCD'])
            # .apply(highlight_value_greater, cols_to_sum=['Unique Subjects'], threshold=.25, subset=['-35 < ITA <= 25 & Monk EFG'])
            # .apply(highlight_value_greater, cols_to_sum=['Unique Subjects'], threshold=.25, subset=['ITA <= -35 & Monk HIJ'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['ITA > 25 & Monk ABCD'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['-35 < ITA <= 25 & Monk EFG'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['ITA <= -35 & Monk HIJ'])

            # Highlight if >=1 subject in category MST 1-4 with ITA >= 50° (monk_forehead & ita_dorsal)
            .map(lambda x: 'background-color: #b5e7a0' if x>=1 else "", subset=['ITA >= 50 & Monk ABCD'])

            # Highlight if >=2 subjects in category MST 8-10 with ITA <= -45°  (monk_forehead & ita_dorsal)
            .map(lambda x: 'background-color: #b5e7a0' if x>=2 else "", subset=['ITA <= -45 & Monk HIJ'])

            # Highlight if the number of sessions with >=25% of so2 data points in the 70%-80%, 80%-90%, and 90% above decade respectively is > 24
            .map(lambda x: 'background-color: #b5e7a0' if x>24 else "", subset=['# of Sessions with >=25%\n of SaO2 in 70-80, 80-90, 90-100'])
            
            # .format(lambda x: f'{x:,.2f}', subset=list(column_dict.values())),
            .format(lambda x: f'{x:,.0f}', subset=list(column_dict_db_new_v1.values())),

            height = (23 + 1) * 35 + 3)
    

if selected_df == "ISO 2024/FDA 2024":
    db_new_v2.rename(columns=column_dict_db_new_v2, inplace=True)
    
    # create a selectbox to choose a monk_forehead
    MST_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    filtered_monk_forehead = ' '
    filtered_monk_forehead = st.selectbox('Select a monk forehead to filter for devices not completed because of it!', [' '] + MST_list)
    if filtered_monk_forehead == ' ':
        pass
    else:
        incomplete_devices = []
        for i in db_new_v2['Device']:
            tdf = db_new_v2[db_new_v2['Device']==i]['Unique Monk Forehead Values']
            if tdf.values[0] == 0 or filtered_monk_forehead not in tdf.values[0]:
                incomplete_devices.append(i)
        db_new_v2_filtered = db_new_v2[db_new_v2['Device'].isin(incomplete_devices)]
        db_new_v2 = db_new_v2_filtered
    
    st.dataframe(db_new_v2
            .style
            # Highlight if Sample size >= 24 (unique patient_id)
            .map(lambda x: 'background-color: #b5e7a0' if x>=24 else "", subset=['Unique Subjects'])

            # Highlight if Average of number of data points per participant/session = 24 (+/-4) (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=20 and x<=28 else "", subset=['Avg Samples per Session'])

            # Highlight if Range of number of data points per participant is in 17-30 (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=24 else "", subset=['Unique Subjects with 17-30 Samples'])

            # Highlight if Each decade between the 70% - 100% saturations contains 33% of the data points (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 70-80 (pooled)'])
            .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 80-90 (pooled)'])
            .map(lambda x: 'background-color: #b5e7a0' if x>= 28 and x<=38 else "", subset=['%\n of SaO2 in 90-100 (pooled)'])
            
            # Highlight if for 70-100% include 73% and 97% (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x<=73 else "", subset=['Min SaO2'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=97 else "", subset=['Max SaO2'])

            # Highlight if >= 90% of the sessions in the same device provide so2 data < 85 (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=90 else "", subset=['%\n of Sessions Provides SaO2 < 85'])

            # Highlight if >=69% participants/sessions provide data points in the 70%-80% decade (sao2)
            .map(lambda x: 'background-color: #b5e7a0' if x>=70 else "", subset=['%\n of Sessions Provides SaO2 in 70-80'])

            # Highlight if Each sex has approximately 33% percentage (assigned_sex)
            .map(lambda x: 'background-color: #b5e7a0' if x>=7.92 else "", subset=['Female'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=7.92 else "", subset=['Male'])

            # Highlight if >= 1 in each of these MST bins (monk_forehead): 1-2, 3-4, 5-6, 7-8, 9-10
            .map(lambda x: 'background-color: #b5e7a0' if x>=1 else "", subset=['Monk AB'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=1 else "", subset=['Monk CD'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=1 else "", subset=['Monk EF'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=1 else "", subset=['Monk GH'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=1 else "", subset=['Monk IJ'])
            
            # Highlight if >= 1 in each of the 10 MST categories (monk_forehead)
            .map(lambda x: 'background-color: #b5e7a0' if x==10 else "", subset=['Unique Monk Forehead'])

            # Highlight if >= 25% in each of the following MST categories: 1-4, 5-7, 8-10 (monk_forehead)
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['Monk ABCD'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['Monk EFG'])
            .map(lambda x: 'background-color: #b5e7a0' if x>=6 else "", subset=['Monk HIJ'])

            # Highlight if the number of sessions with >=25% of so2 data points in the 70%-80%, 80%-90%, and 90% above decade respectively is > 24
            .map(lambda x: 'background-color: #b5e7a0' if x>24 else "", subset=['# of Sessions with >=25%\n of SaO2 in 70-80, 80-90, 90-100'])
            
            # .format(lambda x: f'{x:,.2f}', subset=list(column_dict.values())),
            .format(lambda x: f'{x:,.0f}', subset=list(column_dict_db_new_v2.values())),

            height = (23 + 1) * 35 + 3)             

########### Visualize the skin color distribution of the lab by both ITA and Monk #####################
mscolors = {'A': '#f7ede4', 
            'B': '#f3e7db', 
            'C': '#f6ead0', 
            'D': '#ead9bb', 
            'E': '#d7bd96', 
            'F': '#9f7d54', 
            'G': '#815d44', 
            'H': '#604234', 
            'I': '#3a312a', 
            'J': '#2a2420'}


# import plotly.graph_objects as go

joined_konica_session = create_figure.joined_konica_session(session, konica)

monk_scatter = create_figure.monk_scatter(joined_konica_session, mscolors)

ita_hist = create_figure.ita_hist(joined_konica_session, mscolors)

one, two = st.columns(2)

with one:
    st.plotly_chart(monk_scatter)

with two:
    st.plotly_chart(ita_hist)