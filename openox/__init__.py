from . import colors
from redcap import Project
import pandas as pd
import numpy as np
import streamlit as st
import math
import os
import io
import re
from datetime import datetime, time

def load_project(key):
    """
    Load all data from REDCap using API key specified by 'key'.
    
    Requires a config.py file with a redcap dictionary containing API keys.
    Example of expected dictionary format in config.py: {'REDCAP_SESSION': 'api_key'}
    
    Parameters:
        key (str): The key to access the specific API key from the config file or secrets.

    Returns:
        tuple: A tuple containing the REDCap project instance and a DataFrame of the project records.

    Usage:
        proj_session, session = load_project('session')
    """
    try:
        import config
        api_key = config.redcap[key]
    except:
        print('No config file found')
        try:
            api_key = st.secrets[key]
        except:
            print("Streamlit not installed or no secrets file found")
            raise NameError("No API key files configured")
    api_url = 'https://redcap.ucsf.edu/api/'
    project = Project(api_url, api_key)
    df = project.export_records(format_type='df')
    return project, df

#reshape manual entered data into long format
def reshape_manual(df):
    """
    Reshape manually entered data into long format, suitable for analysis.

    Parameters:
        df (DataFrame): The DataFrame containing the raw data in wide format.

    Returns:
        DataFrame: A DataFrame in long format with columns for saturation, device,
                   probe location, date, session number, and patient ID.

    Usage:
        reshaped_df = reshape_manual(original_df)
    """
    reshaped=pd.DataFrame()

    for index, row in df.iterrows(): #iterate through every patient
        for i in range(1,11): #iterate through each device in every patient
            # create temp df from the row containing only device information
            t2 = row.filter(regex=f'{i}$')
            t2 = pd.DataFrame(t2)

            #label the sample number from the index
            t2['sample_num'] = t2.index
            t2['sample_num'] = t2['sample_num'].str.extract(r'sat(\d+)')

            #within each row, label the device
            t2['device'] = row[f'dev{i}']

            #within each row, label the location
            t2['probe_location'] = row[f'loc{i}']

            #etc
            t2['date'] = row['date']
            t2['session_num'] = row['session']
            t2['patient_id'] = row['patientid']

            #drop the columns not relating to saturation, device and location
            t2 = t2.drop([f'dev{i}', f'loc{i}'])

            #label first column as saturation
            t2.columns.values[0] = 'saturation'

            #concatenate
            reshaped = pd.concat([reshaped, t2], axis=0)

    reshaped=reshaped[reshaped['saturation'].notnull()]
    return reshaped

def ita(row, lab_l, lab_b):
    """
    Calculate the Individual Typology Angle (ITA) from LAB color space values.

    Parameters:
        row (Series): A pandas Series that must include the lab_l and lab_b values.
        lab_l (str): Column name in 'row' that contains the L* value.
        lab_b (str): Column name in 'row' that contains the b* value.

    Returns:
        float: The computed ITA value for the row.

    Usage:
        df['ITA'] = df.apply(ita, lab_l='col_name_labl', lab_b='col_name_lab_b', axis=1)
    """
    return (np.arctan((row[lab_l]-50)/row[lab_b])) * (180/math.pi)

def arms(spo2,so2):
    """
    Calculate the ARMS (Arterial Blood Oxygen Saturation Measurement System) score,
    given paired SpO2 and so2 measurements.

    Parameters:
        spo2 (Series): A pandas Series containing SpO2 values.
        so2 (Series): A pandas Series containing so2 values.

    Returns:
        float: The ARMS score calculated from the mean squared difference between SpO2 and so2.

    Usage:
        masimo_arms = arms(df['Masimo SpO2'], df['so2'])
    """
    return np.sqrt(np.mean((spo2-so2)**2))

def monkcolor(row):
    """
    Recodes 'group' designation to monk site.

    Parameters:
        row (Series): A pandas Series that includes 'group' along with specific anatomical regions.

    Returns:
        str: The value from the designated anatomical region based on group categorization.

    Usage:
        dataframe['color'] = dataframe.apply(monkcolor, axis=1)
    """
    if pd.notna(row['group']):
        if 'Arm' in row['group']:
            return row['monk_upper_arm']
        elif 'Dorsal' in row['group']:
            return row['monk_dorsal']
        elif 'Forehead' in row['group']:
            return row['monk_forehead']
        elif 'Palmar' in row['group']:
            return row['monk_palmar']
        elif 'Fingernail' in row['group']:
            return row['monk_fingernail']


# download labview 2hz files
def get_labview_data(project, record):
    """
    Retrieves LabVIEW data for a specific record from a project.

    Args:
        project (Project): The project object.
        record (str): The record ID.

    Usage:
    labview_list = []
    for record in session['record_id']:
        labview_list.append(get_labview_data(proj_session, record))

    labview_files = pd.concat(labview_list)
    labview_files.reset_index(inplace=True)
    Returns:
        DataFrame: The LabVIEW data as a pandas DataFrame.
    """

    # first get the bytes from the file and put them in a buffer
    f = io.BytesIO(project.export_file(record=record, field='labview_data')[0])

    # Replace non-standard line endings (e.g., \r) with \n
    file_content = f.read().decode('iso-8859-1').replace('\r\n', '\n').replace('\r', '\n')

    # Remove the first two lines of the file
    lines = file_content.split('\n')
    del lines[:2]
    modified_content = '\n'.join(lines).encode('iso-8859-1')
    modified_content = io.BytesIO(modified_content)

    # Read the column names without loading the entire file so we can exclude Raw columns
    column_names = pd.read_table(modified_content, na_values=['---'], nrows=0, encoding='iso-8859-1').columns.tolist()
    filtered_columns = [col for col in column_names if 'Raw' not in col]
    modified_content.seek(0)

    print('Loading LabVIEW data for record {}...'.format(record))
    df = pd.read_table(modified_content, na_values=['---'], usecols=filtered_columns, on_bad_lines='skip', encoding='iso-8859-1')

    # If timestamp is malformed, it will be set to NA
    df['Time Computer'] = pd.to_datetime(df['Time Computer'], errors='coerce')
    df['Timestamp'] = df['Time Computer'].dt.time

    # Filter out any values in the Sample column that are not the string 'Sample #' followed by a number
    # Also replace the string 'Sample #' with an empty string
    pattern = r'^Sample #([0-9]+)$'
    df['Sample'] = df['Sample'].apply(lambda x: re.match(pattern, str(x)).group(1) if pd.notna(x) and re.match(pattern, str(x)) else '')

    raw_columns = [x for x in df.columns if 'Raw' in x]
    df = df.drop(columns=['Time Computer', 'Date Computer', 'Comments'] + raw_columns)

    # Remove rows that don't have an ETCO2, since these are extraneous rows
    # Some contain dates or other malformed information, so this helps to get rid of some of the dates
    df['ETCO2'] = pd.to_numeric(df['ETCO2'], errors='coerce')
    df = df.loc[df.ETCO2.notnull()]

    # Replace double spaces in column names
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

    # Coerce columns to formats
    for col in [
            'Sample',
            'ETCO2',
            'ETO2',
            'ScalcO2',
            'RR',
            'Masimo 97/SpO2',
            'Masimo 97/HR',
            'Masimo 97/PI',
            'Masimo HB/SpO2',
            'Masimo HB/HR',
            'Masimo HB/PI',
            'Nellcor/SpO2',
            'Nellcor/HR',
            'Nellcor/PI',
            'Nihon Koden/SpO2',
            'Nihon Koden/HR',
            'Nihon Koden/PI']:
        if col in df:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.\d+|\d+)')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # add session ID to each row
    df['Session'] = record

    return df

############################################# CLEANING FUNCTIONS ##############################
"""
ABG: 
load abg, threesamples()

LABVIEW:
get_labview_files()
extract_values_by_sample()

base_path = '../OpenOxPHIData/waveforms/labview_2hz/'
pattern = 'labview_session_{}_2hz.csv'
cols_to_summarize = ['Masimo 97/SpO2','Nellcor/SpO2'] 
labview_files = ox.get_labview_files(abg, base_path, pattern)
labview_samples = ox.extract_values_by_sample(labview_files, 'session','Sample',5,cols_to_summarize,abg)

"""

def get_labview_files(abg, base_path, filename_pattern):
    """
    Load files corresponding to each unique session in the ABG data based on a provided filename pattern.
    
    Args:
        abg (pandas.DataFrame): DataFrame containing 'session' column to identify unique sessions.
        base_path (str): The base directory where the files are stored.
        filename_pattern (str): A pattern to format filenames with session numbers. 
                                It should include `{}` where the session number should be inserted.
    
    Returns:
        pandas.DataFrame: A DataFrame containing concatenated data from all available session files.
    """
    labview_files = []

    for i in abg['session'].unique().astype(int):
        # Format the filename according to the pattern provided
        file_path = os.path.join(base_path, filename_pattern.format(i))
        try:
            tdf = pd.read_csv(file_path)
            tdf['session'] = i
            labview_files.append(tdf)
        except FileNotFoundError:
            print(f'No file for session {i} at {file_path}')
        except Exception as e:
            print(f'Error loading file for session {i}: {e}')

    if labview_files:
        labview_files = pd.concat(labview_files, ignore_index=True)
    else:
        labview_files = pd.DataFrame()  # Return an empty DataFrame if no files were loaded

    return labview_files

def extract_values_by_sample(df, sessioncol, samplecol, average_over, cols_to_summarize, abg):
    """
    Extracts values from a DataFrame by sample, averaging over a specified time period.
    
    Args:
        df (pandas.DataFrame): The input DataFrame, a concatenation of all the relevant 2Hz session files.
        sessioncol (str): The name of the column representing the session.
        samplecol (str): The name of the column representing the sample.
        average_over (int): The time period (in seconds) over which to average the values.
        cols_to_summarize (list): The list of column names to extract and average.
        abg (pandas.DataFrame): DataFrame containing ABG data to merge with the results.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the extracted values, averaged over the specified time period for each sample.
    """
    print('extracting labview values.....')
    results = pd.DataFrame(columns=[sessioncol, samplecol] + cols_to_summarize + ['Timestamp'])
    for session_id, session in df.groupby(sessioncol):
        for i in session[samplecol].dropna().unique():
            start_row = session[session[samplecol] == i].index[-1]
            end_row = start_row + average_over * 2

            nextrow = [session_id, i]
            for col in cols_to_summarize:
                nextrow.append(session.loc[start_row:end_row, col].mean())
            nextrow.append(session.loc[start_row:end_row, 'Timestamp'].max())
            
            results.loc[len(results)] = nextrow
    if 'Sample' in results.columns:
        results.rename(columns={'Sample': 'sample'}, inplace=True)

    # now we merge with abg, and groupby to get rid of duplicate 'samples' when the button is held
    results['sample'] = results['sample'].astype(int)
    results['session'] = results['session'].astype(int)
    results = results.merge(abg, on=['session', 'sample'], how='inner')
    results = results.groupby(['session', 'sample']).agg({col: 'mean' for col in results.select_dtypes(include=[np.number]).columns if col != 'Timestamp'}).assign(Timestamp=lambda x: results.groupby(['session', 'sample'])['Timestamp'].last()).drop(['session', 'sample'], axis=1).reset_index()

    return results

import pandas as pd

def sample_stability_multi(df, so2_col, nellcor_col, timestamp_col, output_newcol_name, so2_bound, ref_bound):
    """
    Checks the stability of so2 and Nellcor samples in the DataFrame, rejecting unstable readings.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing so2 and Nellcor readings.
        so2_col (str): Column name for so2 readings.
        nellcor_col (str): Column name for Nellcor readings.
        timestamp_col (str): Column name for timestamp data.
        output_newcol_name (str): Name of the new column to add the decisions.
        bound (int): Threshold for deciding if a sample is stable.
    
    Returns:
        pandas.DataFrame: The DataFrame with an additional column indicating sample stability.
    """
    # Convert timestamp to datetime if not already
    df=df.sort_values(by=['session','sample'], ascending=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

    # Calculate differences using shift for both so2 and Nellcor, and handle timestamp
    for col in ['sample',so2_col, nellcor_col, timestamp_col]:
        if col == timestamp_col:
            # Calculate time difference in seconds
            df[f'{col}_diff_prev'] = (df[col] - df[col].shift(1)).dt.total_seconds()
            df[f'{col}_diff_next'] = (df[col] - df[col].shift(-1)).dt.total_seconds()
        else:
            df[f'{col}_diff_prev'] = df[col] - df[col].shift(1)
            df[f'{col}_diff_next'] = df[col] - df[col].shift(-1)
        # >> Fill NaN values with 0 for the first samples because they dont have a previous time.
        df.loc[df['sample'] == 1, f'{col}_diff_prev'] = df.loc[df['sample'] == 1, f'{col}_diff_prev'].fillna(0) 

    def check_bound(row, check_col, bound):
        if abs(row[f'{check_col}_diff_prev']) >= bound:  # Check previous sample. If outside of bound, check next sample
            if abs(row[f'{check_col}_diff_next']) >= bound:
                return pd.Series([False, 'reject because both previous and next samples are outside bound'])
            else:
                return pd.Series([True, 'keep because next sample is within bound'])
        else:
            return pd.Series([True, 'keep because previous sample is within bound'])

    # Apply the function and assign results to two new columns
    df[['so2_stable', 'so2_reason']] = df.apply(check_bound, axis=1, check_col=so2_col, bound=so2_bound)

    # If you also need to apply it for 'Nellcor', repeat with the appropriate column names
    df[['Nellcor_stable', 'Nellcor_reason']] = df.apply(check_bound, axis=1, check_col=nellcor_col, bound=ref_bound)

    # Combine stability decisions
    # df[output_newcol_name] = ((df['so2_stable'] & df['Nellcor_stable'])).replace({True: 'keep', False: 'reject'})
    # label output_newcol_name based on combination of so2 and nellcor stability. if both kept, then keep. if so2 kept but nellcor rejected, then 'reject_nellcor'. if so2 rejected but nellcor kept, then 'reject_so2'. if both rejected, then 'reject_both'
    df[output_newcol_name+'_status'] = ((df['so2_stable'] & df['Nellcor_stable']))
    df[output_newcol_name] = ((df['so2_stable'] & df['Nellcor_stable'])).replace({True: 'keep', False: 'reject_both'})   
    df.loc[(df['so2_stable'] & ~df['Nellcor_stable']), output_newcol_name] = 'reject_nellcor'
    df.loc[(~df['so2_stable'] & df['Nellcor_stable']), output_newcol_name] = 'reject_so2'


    
    return df, df[output_newcol_name].value_counts().to_dict()

def session_criteria_check(df):
    """
    Checks session criteria:
    1. at least one so2 data point in 97-100 range;
    2. at least one so2 data point in 67-73 range;
    3. all plateaus have at least 6 so2 data points;
    
    Args:
        df (pandas.DataFrame): The DataFrame displayed in Algo comparison page.
    
    Returns:
        criteria_check_tuple: A tuple containing the True/False results of the 3 session criteria checks.
        criteria_check_df: A DataFrame contains more info about session check and is displayed on dashboard.
    """    
    criteria_check_tuple = (df[(df['so2'] >= 97) & (df['so2'] <= 100)].shape[0] > 0,
                      df[(df['so2'] >= 67) & (df['so2'] <= 73)].shape[0] > 0,
                      (df[(df['so2'] >= 70) & (df['so2'] <= 80)].shape[0] >= 6) and 
                      (df[(df['so2'] >= 80) & (df['so2'] <= 90)].shape[0] >= 6) and
                      (df[(df['so2'] >= 90) & (df['so2'] <= 100)].shape[0] >= 6))
    criteria_check_df = pd.DataFrame({'#so2 in 97-100': [df[(df['so2'] >= 97) & (df['so2'] <= 100)].shape[0]],
                                      '#so2 in 67-73': [df[(df['so2'] >= 67) & (df['so2'] <= 73)].shape[0]],
                                      '#so2 in 70-80': [df[(df['so2'] >= 70) & (df['so2'] <= 80)].shape[0]],
                                      '#so2 in 80-90': [df[(df['so2'] >= 80) & (df['so2'] <= 90)].shape[0]],
                                      '#so2 in 90-100': [df[(df['so2'] >= 90) & (df['so2'] <= 100)].shape[0]]})
    return criteria_check_tuple, criteria_check_df