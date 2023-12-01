import hypoxialab_functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_figures(konica, session, fig_size):
    
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
    
    ############# scatterplot #############
    sns.set_theme(rc={'figure.dpi':150, 'figure.figsize':fig_size}, style='whitegrid', font_scale=1.0)
    joined_konica_session.sort_values(by='monk', inplace=True)

    # extend the x-axis labels to include 'J'
    x_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    sns.scatterplot(data = joined_konica_session, x='monk', y='ita', hue='monk', palette=mscolors, edgecolor='black', linewidth=0.8,)
    # set x-axis labels
    plt.xticks(range(len(x_labels)), x_labels)

    plt.gca().set_title('Hypoxia Lab: Monk scale vs ITA measurement')
    plt.gca().set_xlabel('Monk')
    plt.gca().set_ylabel('ITA')
    #hide legend
    plt.gca().get_legend().remove()

    sns.despine(left=True, bottom=True)
    
    ############### barplot ###############
    # Create a list of unique sorted values in the 'Monk' column
    unique_monk_values = sorted(joined_konica_session['monk'].unique())

    # Add 'J' to the unique values
    unique_monk_values.append('J')

    # Create a new palette with colors for all unique values
    new_palette = [mscolors[monk] for monk in unique_monk_values]

    sns.set_theme(rc={'figure.dpi': 150, 'figure.figsize': fig_size}, style='whitegrid', font_scale=1.0)

    def monkrgb(row):
        if row['monk'] in mscolors:
            return mscolors[row['monk']]

    joined_konica_session['monkrgb'] = joined_konica_session.apply(monkrgb, axis=1)
    joined_konica_session.sort_values(by='monk', inplace=True)

    fig, ax = plt.subplots()

    sns.histplot(data=joined_konica_session, x='ita', hue='monk', palette=new_palette, multiple='stack', bins=20)

    sns.despine(left=True, bottom=False)
    ax.set(xlabel='ITA', ylabel='Count', title='Hypoxia Lab ITA Distribution')

    # Manually create a custom legend
    legend_labels = unique_monk_values
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mscolors[monk], markersize=10) for monk in unique_monk_values]

    ax.legend(legend_handles, legend_labels, title='Monk')
    
    # flip the x-axis
    ax.invert_xaxis()
    # plt.savefig('hl_ita_hist.png', dpi=300)
    # plt.show()
    
    return plt.gcf(), fig