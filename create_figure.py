import hypoxialab_functions
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def monk_scatter(konica, session):
    
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
    
    ########## Creating the figures #############
    mscolors = {'A': '#f7ede4', 'B': '#f3e7db', 'C': '#f6ead0', 'D': '#ead9bb', 'E': '#d7bd96', 'F': '#9f7d54', 'G': '#815d44', 'H': '#604234', 'I': '#3a312a', 'J': '#2a2420'}
    
    # Specify the desired order for the labels
    label_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    # Set the category order for the x-axis labels
    xaxis = dict(categoryorder='array', categoryarray=label_order, gridcolor='lightgrey')

    # Create a scatter plot with individual traces
    traces = []
    for label in label_order:
        filtered_data = joined_konica_session[joined_konica_session['monk'] == label]
        trace = go.Scatter(
            x=filtered_data['monk'],
            y=filtered_data['ita'],
            mode='markers',
            name=label,
            marker=dict(color=mscolors[label], line=dict(color='black', width=1)))
        traces.append(trace)

    # Create a dummy trace for 'J'
    trace_J = go.Scatter(
        x=['J'],
        y=[None],
        mode='markers',
        name='J',
        marker=dict(color=mscolors['J'], line=dict(color='black', width=1))  # Set marker border color to black
    )

    # Add the dummy trace for 'J' to the list of traces
    traces.append(trace_J)

    # Create the layout with the x-axis category order
    layout = go.Layout(
        xaxis = xaxis,
        xaxis_title = 'Monk',
        yaxis = dict(
            title = 'ITA',
            range = [-80, 80],
            dtick = 20,
            gridcolor = 'lightgrey',
            zerolinecolor = 'lightgrey',
            zerolinewidth = 1
        ),  
        title = 'Monk vs ITA by Monk Color', 
        legend_title = 'Monk',
        paper_bgcolor = 'white',  # Set the background color for the entire plot
        plot_bgcolor = 'white'   # Set the background color for the plot area  
    )

    # Create the figure
    monk_scatter = go.Figure(data=traces, layout=layout)
    
    return monk_scatter

def ita_hist(konica, session):
    
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
    
    ########## Creating the figures #############
    mscolors = {'A': '#f7ede4', 'B': '#f3e7db', 'C': '#f6ead0', 'D': '#ead9bb', 'E': '#d7bd96', 'F': '#9f7d54', 'G': '#815d44', 'H': '#604234', 'I': '#3a312a', 'J': '#2a2420'}
    
    ita_hist = px.histogram(joined_konica_session, x='ita', title='ITA Distribution by Monk Color',
                        color = 'monk', # I don't know why it's not mapping correctly to colors but we can work on this later...
                        color_discrete_map=mscolors).update_xaxes(title_text='ITA').update_yaxes(title_text='Count')
    
    return ita_hist