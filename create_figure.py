import hypoxialab_functions
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def joined_konica_session(session, konica):
    ########### Get the konica data #############
    konica_sub = konica[['upi', 'session', 'group', 'lab_l', 'lab_a', 'lab_b']].rename(columns={'upi': 'patient_id'})
    # take the median of each unique session
    konica_sub = konica_sub.groupby(['session', 'group']).median(numeric_only=True).reset_index()
    # calculate the ITA
    konica_sub['ita'] = konica_sub.apply(hypoxialab_functions.ita, axis=1)
    konica_sub = konica_sub.drop(['lab_l', 'lab_a', 'lab_b'], axis=1)
    konica_sub = konica_sub.dropna(subset=['ita'])

    ########### Get the session data #############
    session_sub = session[['patient_id', 'record_id'] + [x for x in session.columns if x.startswith('monk')]].rename(columns={'record_id': 'session'})
    # drop the na 
    session_sub = session_sub.dropna(subset=['monk_fingernail', 'monk_dorsal', 'monk_palmar', 'monk_upper_arm', 'monk_forehead'], how='all')

    ########### Merge the two tables #############
    joined_konica_session = pd.merge(session_sub, konica_sub, on='session', how='left')

    # fix typos in 'group'
    joined_konica_session['group'] = joined_konica_session['group'].str.replace('Inner Arn (D)', 'Inner Upper Arm (D)')
    joined_konica_session['group'] = joined_konica_session['group'].str.replace('Dorsal - DIP (B)', 'Dorsal (B)')
    joined_konica_session['group'] = joined_konica_session['group'].str.replace('Forehead (G)', 'Forehead (E)')

    joined_konica_session['monk'] = joined_konica_session.apply(hypoxialab_functions.monkcolor, axis=1)
    joined_konica_session = joined_konica_session.drop(['monk_fingernail', 'monk_dorsal', 'monk_upper_arm', 'monk_forehead', 'monk_palmar'], axis=1)

    joined_konica_session = joined_konica_session.dropna(subset=['monk', 'ita']).drop(['patient_id_y', 'patient_id_x'], axis=1)
    return joined_konica_session


def monk_scatter(joined_konica_session, mscolors):
    
    monk_scatter = px.scatter(joined_konica_session.sort_values(by='monk'), x='monk', y='ita', 
                color='monk', 
                title='Monk vs ITA by Monk Color',
                color_discrete_map=mscolors,
                labels={"monk": "Monk"}).update_xaxes(title_text='Monk').update_yaxes(title_text='ITA', range=[-80, 80], dtick=20).update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                
    monk_scatter.add_trace(go.Scatter(x=['J'], y=[np.nan], name='J')) # if you insist on including the blank 'J' column :)
    
    return monk_scatter

def ita_hist(joined_konica_session, mscolors):
    
    # Sort the data based on the order of keys in mscolors -> so the color can be mapped correctly?
    joined_konica_session['monk'] = joined_konica_session['monk'].astype('category')
    joined_konica_session['monk'].cat.set_categories(list(mscolors.keys()))
    joined_konica_session = joined_konica_session.sort_values('monk', ascending=False)

    ita_hist = px.histogram(joined_konica_session, x='ita', title='ITA Distribution by Monk Color',
                        color = 'monk', # I don't know why it's not mapping correctly to colors but we can work on this later...
                        color_discrete_map = mscolors,
                        nbins = 20
                        ).update_xaxes(title_text='ITA',range=[80, -80], dtick=20).update_yaxes(title_text='Count')

    # make the legend label in correct order
    ita_hist.update_layout(
        legend_title_text='Monk',
        legend_traceorder="reversed"
    )
    
    # ita_hist = px.histogram(joined_konica_session, x='ita', title='ITA Distribution by Monk Color',
    #                     color = 'monk', # I don't know why it's not mapping correctly to colors but we can work on this later...
    #                     color_discrete_map=mscolors).update_xaxes(title_text='ITA').update_yaxes(title_text='Count')
    
    return ita_hist