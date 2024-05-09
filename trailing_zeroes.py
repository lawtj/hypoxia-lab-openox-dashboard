import numpy as np
import pandas as pd
import plotly.express as px

# this function recalculates the so2 range and can be run each time after an ABG fix has been applied.
def recalculate_so2_range(df, encountercol, so2thresh):
    if 'so2_range' in df.columns:
        df.drop('so2_range', axis=1, inplace=True)
    ###---------------------------------------------------------------------------------------------------------------------------------------------------------###
    for encounter in df['session'].unique():
        df_encounter = df[df['session'] == encounter]
        for sample in df_encounter['sample'].unique():
            df_encounter_sample = df_encounter[df_encounter['sample'] == sample].copy()
            
            ## --------------- calculate the difference between the median and the so2 value here
            median_so2 = df_encounter_sample['so2'].median()
            df_encounter_sample['diff'] = df_encounter_sample['so2'] - median_so2
            ## --------------- if any difference greater than 1.5, we need to skip this sample - might be a trailing 0s problem or something else weird
            ## we comment out this part of the code now since we start reviewing session in the QA script 
            # if len(df_encounter_sample[df_encounter_sample['diff'].abs() >= 1.5]) > 0:
            #     df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
            #     continue
            
            # now we skip those samples that have so2 values that are more than 1.5 apart
            if len(df_encounter_sample) == 2:
                df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
            elif len(df_encounter_sample) > 2:
                # if we do have two so2 values that are no more than 0.5 apart, drop the row where the difference is greater than 0.5
                if len(df_encounter_sample[df_encounter_sample['diff'].abs() <= 0.5]) > 0:
                    df.drop(df_encounter_sample[df_encounter_sample['diff'].abs() > 0.5].index, inplace=True)
                    # now calculate the difference between the highest and lowest so2 values
                    df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
                else:
                    df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
    ###---------------------------------------------------------------------------------------------------------------------------------------------------------###
    
    # so2_range = df.groupby([encountercol,'sample'])['so2'].agg(so2_range=lambda x: x.max() - x.min()).reset_index()
    # df = df.merge(so2_range, on=[encountercol,'sample'])
    so2_count = len(df[abs(df['so2_range']) > so2thresh])
    fig = px.scatter(df, x='sample', y='so2_range', color='sample', template='plotly_white', hover_data=['so2','so2_range','sample','session']) 
    fig.update_layout(title='So2 Range by Sample: Number of so2 data points that has range > {} = {}'.format(so2thresh, so2_count))
    return fig


def fix_trailing_zeroes(abg, primary_thresh):
    # list of sessions where abg timestamps are on two different days
    sessioncount = (abg.groupby(['session'])['date_calc'].nunique())
    sessions_with_one_date = sessioncount[sessioncount == 1]
    sessions_with_one_date = sessions_with_one_date.index.tolist()
    sessions_with_multiple_dates = sessioncount[sessioncount > 1]
    sessions_with_multiple_dates = sessions_with_multiple_dates.index.tolist()

    df_time_index = abg.groupby(['session','sample'])['time_calc'].min().reset_index()
    for session, encounter_group in abg[abg['session'].isin(sessions_with_one_date)].groupby(['session']):
        primary_thresh_time = None
        try:
            primary_thresh_time = encounter_group[encounter_group['sample']==primary_thresh].sort_values(by='time_calc')['time_calc'].values[0]
        except:
            print(session, 'does not have a sample', primary_thresh)

        # check to see if sample X exists
        if primary_thresh_time is not None:
            m1 = encounter_group['sample']==1
            m2 = encounter_group['time_calc'] >= primary_thresh_time
            mindex = encounter_group[m1 & m2].index
            
            abg.loc[mindex,'sample'] = 10
            
            # if sample 2 is after sample 9, then change it to 20
            m1 = encounter_group['sample']==2
            mindex = encounter_group[m1 & m2].index
            abg.loc[mindex,'sample'] = 20

            # if sample 3 is after sample 9, then change it to 30
            m1 = encounter_group['sample']==3
            mindex = encounter_group[m1 & m2].index
            abg.loc[mindex,'sample'] = 30

    return abg, sessions_with_multiple_dates

def find_distance(row, s1max, s1min, s2max, s2min):
    if (row['sample'] == 1):
        if s1min == None:
            return row['sample']
        else:
            distance_s1 = abs(row['so2'] - s1max)
            distance_s1min = abs(row['so2'] - s1min)
            return 1 if distance_s1 < distance_s1min else 10
    if (row['sample'] == 2):
        if s2min == None:
            return row['sample']
        distance_s2 = abs(row['so2'] - s2max)
        distance_s2min = abs(row['so2'] - s2min)
        return 2 if distance_s2 < distance_s2min else 20
    else:
        return row['sample']

def fix_trailing_zeroes_nearest_neighbor(abg, sessions_with_multiple_dates):
    print('trimming trailing zeroes')
    # Iterate over each unique session number and its corresponding group in the DataFrame 'abg'
    for session_num, session_group in abg[abg['session'].isin(sessions_with_multiple_dates)].groupby(['session']):
        
        # Calculate the median values for each unique 'sample' within the session group
        sampleindex = session_group.groupby('sample').median(numeric_only=True).reset_index()
        
        # Initialize variables for storing maximum and minimum values
        s1max = 100
        s1min = s2max = s2min = None

        # Check if sample 11 exists in the 'sampleindex' DataFrame
        if 11 in sampleindex['sample'].values:
            # If sample 11 exists, retrieve its corresponding 'so2' value and assign it to 's1min'
            s1min = sampleindex[sampleindex['sample'] == 11]['so2'].values[0]
            # check if sample 9 also exists, if so, make s1min the average of the two values in case sample 11 is at a lower plateau
            if 9 in sampleindex['sample'].values:
                s1min = np.mean([sampleindex[sampleindex['sample'] == 9]['so2'].values[0], s1min])
                    
        # Check if sample 4 exists in the 'sampleindex' DataFrame
        if 4 in sampleindex['sample'].values:
            # If sample 4 exists, retrieve its corresponding 'so2' value and assign it to 's2max'
            s2max = sampleindex[sampleindex['sample'] == 4]['so2'].values[0]
        
        # Check if sample 19 exists in the 'sampleindex' DataFrame
        if 19 in sampleindex['sample'].values:
            # If sample 19 exists, retrieve its corresponding 'so2' value and assign it to 's2min'
            s2min = sampleindex[sampleindex['sample'] == 19]['so2'].values[0]
            # check if sample 21 also exists, if so, make s2min the average of the two values in case sample 19 is at a higher plateau
            if 21 in sampleindex['sample'].values:
                s2min = np.mean([sampleindex[sampleindex['sample'] == 21]['so2'].values[0], s2min])


        # Print the current session number
        # print('starting session', session_num)
        
        # Apply the 'find_distance' function to each row in the session group, passing the calculated values as arguments
        updated_samples = session_group.apply(find_distance, args=(s1max, s1min, s2max, s2min), axis=1)

        # Update the 'sample' column in the original DataFrame 'abg' with the updated samples
        abg.loc[session_group.index, 'sample'] = updated_samples

    ###---------------------------------------------------------------------------------------------------------------------------------------------------------###

def threesamples(abg):
    ## goal: if the CRC was doing 3 runs for the blood sample, throw out the so2 value that is off
    # group by encountercol and sample and count the number of so2 values
    # if there are only two so2 values, then the so2 range is the difference between the two so2 values
    # else if there are more than two samples, calculate the so2 range for the two samples that has so2 value less than 0.5 apart.
    # else if there are no samples that are less than 0.5 apart, then the so2 range is the difference between the highest and lowest so2 values - werid if has
    exclude_list = []

    for group, sample in abg.groupby(['session','sample']):
        #group is a tuple of (current session, current sample), and sample is the dataframe filtered 
            ## --------------- calculate the difference between the median and the so2 value here
            median_so2 = sample['so2'].median()
            sample['diff'] = sample['so2'] - median_so2
            ## --------------- if any difference greater than 1.5, we need to skip this sample - might be a trailing 0s problem or something else weird
            # if len(sample[sample['diff'].abs() >= 1.5]) > 0:
            #     # print(group[0])
            #     abg.loc[(abg['session'] == group[0]) & (abg['sample'] == group[1]), 'so2_range'] = sample['so2'].max() - sample['so2'].min()
            #     continue
    
            # now we've skipped those samples that have so2 values that are more than 1.5 apart
            if len(sample) == 2:
                # if the two so2 values are more than 1 apart, then we need to exclude this sample
                if sample['so2'].max() - sample['so2'].min() > 1:
                    exclude_list.append([group[0], group[1], sample['so2'].max(), sample['so2'].min(), sample['so2'].max() - sample['so2'].min()])
                abg.loc[(abg['session'] == group[0]) & (abg['sample'] == group[1]), 'so2_range'] = sample['so2'].max() - sample['so2'].min()

            elif len(sample) > 2:
                try:
                    # if none of the samples are less than 0.5 apart, then we need to exclude this sample
                    if len(sample[sample['diff'].abs() <= 0.5]) < 2:
                        abg.drop(abg[(abg['session'] == group[0]) & (abg['sample'] == group[1])].index, inplace=True)
                    else:
                        # if we do have two so2 values that are no more than 0.5 apart, drop the row where the difference is greater than 0.5
                        abg.drop(sample[sample['diff'].abs() > 0.5].index, inplace=True)
                        # now calculate the difference between the highest and lowest so2 values
                        abg.loc[(abg['session'] == group[0]) & (abg['sample'] == group[1]), 'so2_range'] = sample['so2'].max() - sample['so2'].min()
                except Exception as e:
                    print('Exception!: ', e)

    # ###---------------------------------------------------------------------------------------------------------------------------------------------------------###
    # drop rows of where session and sample is in the exclude list
    for row in exclude_list:
        abg.drop(abg[(abg['session'] == row[0]) & (abg['sample'] == row[1])].index, inplace=True)