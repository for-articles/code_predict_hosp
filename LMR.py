
# extract 'koblingsnoekkel' in lMR that has no corresponding 'lopenr' in NPR files in file "missing_koblingsnoekkel.dta" (done) 
# append LMR files (done)
# remove these patients in "missing_koblingsnoekkel.dta" from LMR dataset (done)
# convert the dataframe from .csv to .dta in stata to avoid the dates foramt conflict
# make ttt episods

# Filtering out patient that are only in LMR
import pandas as pd
import glob
import os

# Collect unique 'koblingsnoekkel' from LMR_* files
directory = r'archive/Masterfiler for Mohsen/' 
lmr_file_pattern = os.path.join(directory, "LMR_*.dta")
lmr_files = glob.glob(lmr_file_pattern)
all_koblingsnoekkel = set()

# Iterate over each LMR file and collect 'koblingsnoekkel'
for file in lmr_files:
    df = pd.read_stata(file)
    all_koblingsnoekkel.update(df['koblingsnoekkel'].unique())

# Collect unique 'lopenr' from aktivitet_* files
npr_file_pattern = os.path.join(directory, "aktivitet_*.dta")
npr_files = glob.glob(npr_file_pattern)
all_lopenr = set()

# Iterate over each NPR file and collect 'lopenr'
for file in npr_files:
    df = pd.read_stata(file)
    all_lopenr.update(df['lopenr'].unique())

# Find 'koblingsnoekkel' values that do not match with 'lopenr'
missing_koblingsnoekkel = all_koblingsnoekkel.difference(all_lopenr)

# Create a DataFrame from the missing 'koblingsnoekkel'
missing_koblingsnoekkel_df = pd.DataFrame(list(missing_koblingsnoekkel), columns=['koblingsnoekkel'])

# the DataFrame as a .dta file
output_file = r'missing_koblingsnoekkel.dta'
missing_koblingsnoekkel_df.to_stata(output_file, write_index=False)

print(f"File saved: {output_file}. Number of missing 'koblingsnoekkel': {len(missing_koblingsnoekkel)}")
import pandas as pd
lopenr_LMR = pd.read_stata(r'missing_koblingsnoekkel.dta')

lopenr_LMR.shape #ok


# append LMR files
import pandas as pd
import glob
import os

# Step 1: Define the directory and pattern to match all LMR files
directory = r'archive/Masterfiler for Mohsen/' 
lmr_file_pattern = os.path.join(directory, "LMR_*.dta")
lmr_files = glob.glob(lmr_file_pattern)

# Step 2: Initialize an empty list to store the DataFrames
lmr_dataframes = []

# Step 3: Load each LMR file and append it to the list
for file in lmr_files:
    df = pd.read_stata(file)
    lmr_dataframes.append(df)

# Step 4: Concatenate all DataFrames into one DataFrame
combined_lmr_df = pd.concat(lmr_dataframes, ignore_index=True)

# Step 5: Save the combined DataFrame as a CSV file
output_csv_file = r'LMR_data_all.csv'
combined_lmr_df.to_csv(output_csv_file, index=False)

print(f"Combined LMR files have been saved to: {output_csv_file}")

import pandas as pd
df_LMR_All=pd.read_csv(r'LMR_data_all.csv') # before removing the LMR only patients

df_LMR_All['koblingsnoekkel'].nunique()



# remove patients that only exist in LMR and not NPR
import pandas as pd

# Step 1: Load the combined LMR file (CSV)
combined_lmr_df = pd.read_csv(r'LMR_data_all.csv')

# Step 2: Load the missing koblingsnoekkel file (from the .dta file)
missing_koblingsnoekkel_df = pd.read_stata(r'missing_koblingsnoekkel.dta')

# Step 3: Remove rows where 'koblingsnoekkel' in combined_lmr_df is present in missing_koblingsnoekkel_df
filtered_lmr_df = combined_lmr_df[~combined_lmr_df['koblingsnoekkel'].isin(missing_koblingsnoekkel_df['koblingsnoekkel'])]

# Step 4: Save the filtered DataFrame to a new CSV file
output_filtered_csv = r'LMR_data_all.csv'
filtered_lmr_df.to_csv(output_filtered_csv, index=False)

print(f"Filtered LMR files have been saved to: {output_filtered_csv}")

import pandas as pd
df_LMR_alone=pd.read_csv(r'LMR_data_all.csv') # after removing the LMR only patients
df_LMR_alone['koblingsnoekkel'].nunique()

51665-3261 #ok

# rename 'koblingsnoekkel' --> 'lopenr' 
df_LMR_alone = df_LMR_alone.rename(columns={'koblingsnoekkel': 'lopenr'})

# save
df_LMR_alone.to_csv(r'LMR_data_all.csv', index=False)
df_LMR_alone.columns

# Display max rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import pandas as pd
df=pd.read_csv(r'LMR_data_all.csv')
df.head()

import pandas as pd
import numpy as np
# making ttt episode 

# calculate the cohort duration of each patient in the dataset
# Convert 'dato' to datetime format
df['utlevering_dato'] = pd.to_datetime(df['utlevering_dato'])
# Groupby lopnr
patient_durations = df.groupby('lopenr')['utlevering_dato'].agg([min, max])

# Calculate the duration difference between the last and first prescription date
patient_durations['duration'] = (patient_durations['max'] - patient_durations['min']).dt.days

# Reset index to make 'lopenr' a column again
patient_durations.reset_index(inplace=True)
patient_durations.head()
# save 
df = pd.merge(df, patient_durations[['lopenr', 'duration']], on='lopenr', how='left')

# Rename some columns to match the code (should be renamed back later)
df.rename(columns={
    'legemiddel_atckode_niva5': 'atckode',
    'lopenr': 'pasientlopenr',
    'utlevering_dato': 'diff_utleveringdato',
    'utlevering_ddd': 'ordinasjonantallddd'
}, inplace=True)


df[['pasientlopenr', 'atckode', 'diff_utleveringdato', 'ordinasjonantallddd']].head(), df.dtypes[['pasientlopenr', 'atckode', 'diff_utleveringdato', 'ordinasjonantallddd']]
df.head()

# Check for conversion issues (e.g., missing or invalid dates)
print(df['diff_utleveringdato'].isna().sum())  # Shows how many NaT values there are

# Convert to datetime if not done already
df['diff_utleveringdato'] = pd.to_datetime(df['diff_utleveringdato'], errors='coerce')
df['ordinasjonantallddd'] = df['ordinasjonantallddd'].str.replace(',', '.', regex=False)
df['utlevering_antallpakninger'] = df['utlevering_antallpakninger'].str.replace(',', '.', regex=False)
df['vare_ddd_verdi'] = df['vare_ddd_verdi'].str.replace(',', '.', regex=False)
df.head()

# Convert to numeric, coercing errors to NaN
df['ordinasjonantallddd'] = pd.to_numeric(df['ordinasjonantallddd'], errors='coerce')


# Check how many non-numeric values were coerced to NaN
df['ordinasjonantallddd'].head()

# step 1.Summation and Preparation:
# summing 'ordinasjonantallddd'
df['sum_ordinasjonantallddd'] = df.groupby(['pasientlopenr', 'atckode', 'diff_utleveringdato'])['ordinasjonantallddd'].transform('sum')
print(df['sum_ordinasjonantallddd'].sum()) #ok

# Remove duplicates
df.drop_duplicates(subset=['pasientlopenr', 'atckode', 'diff_utleveringdato', 'ordinasjonantallddd'], inplace=True)

# sorting
df = df.sort_values(by=['pasientlopenr', 'atckode', 'diff_utleveringdato'])

# Lag and Lead Variables:
df['lag_dato'] = df.groupby(['pasientlopenr', 'atckode'])['diff_utleveringdato'].shift(1)
df['lead_dato'] = df.groupby(['pasientlopenr', 'atckode'])['diff_utleveringdato'].shift(-1)

# convert columns to datetime
df['lead_dato'] = pd.to_datetime(df['lead_dato'])
df['diff_utleveringdato'] = pd.to_datetime(df['diff_utleveringdato'])

# calculate the date differences
df['delta_dager'] = df['lead_dato'] - df['diff_utleveringdato']
df['delta_dager'] = df['delta_dager'].dt.days
df['delta_dager'].fillna(0, inplace=True),
df['ordinasjonantallddd'].head()

# Similar operations for DDD variables
df['lag_ddd'] = df.groupby(['pasientlopenr', 'atckode'])['ordinasjonantallddd'].shift(1)
df['lead_ddd'] = df.groupby(['pasientlopenr', 'atckode'])['ordinasjonantallddd'].shift(-1)

# Adherence calculation
df['ordinasjonantallddd'] = pd.to_numeric(df['ordinasjonantallddd'], errors='coerce')
df['ddd_80p_adh'] = df['ordinasjonantallddd'] / 0.8
df['lag_ddd_80p_adh'] = df.groupby(['pasientlopenr', 'atckode'])['ddd_80p_adh'].shift(1)
df['lead_ddd_80p_adh'] = df.groupby(['pasientlopenr', 'atckode'])['ddd_80p_adh'].shift(-1)
# Prescription coverage and carryover
df['resept_dekning_dager'] = df['ddd_80p_adh'] - df['delta_dager']
df['resept_dekning_dager'] = df['resept_dekning_dager'].apply(lambda x: max(x, 0))

df['lag_rx_dekn_dag'] = df.groupby(['pasientlopenr', 'atckode'])['resept_dekning_dager'].shift(1).fillna(0)

df['carryover'] = df['ddd_80p_adh'] - df['delta_dager'] + df['lag_rx_dekn_dag']
df['lag_carryover'] = df.groupby(['pasientlopenr', 'atckode'])['carryover'].shift(1)
df['treatment_episode'] = 0
df.loc[df['lag_carryover'].isna() | (df['lag_carryover'] < -14), 'treatment_episode'] = 1
df.loc[df['carryover'] < -14 | df['lead_dato'].isna(), 'treatment_episode'] = 3

# Set initial values for treatment start and end based on episodes
df['treatment_start'] = np.where(df['treatment_episode'] == 1, df['diff_utleveringdato'], pd.NaT)
df['treatment_end'] = np.where(df['treatment_episode'] == 3, df['lead_dato'], pd.NaT)

# Fill treatment start and end across all episodes
df['treatment_start'] = df.groupby(['pasientlopenr', 'atckode'])['treatment_start'].fillna(method='ffill')
df['treatment_end'] = df.groupby(['pasientlopenr', 'atckode'])['treatment_end'].fillna(method='bfill')
df.head()




# Filter to keep only episodes that end (3)
df = df[df['treatment_episode'] == 3]

# Check results
print(df[['pasientlopenr', 'atckode', 'treatment_episode', 'treatment_start', 'treatment_end']].head())
df['treatment_end'].isna().sum()

# fill ttt start with corresponding prescription date
# Convert diff_utleveringdato to the same integer timestamp format as treatment_start
df['diff_utleveringdato'] = df['diff_utleveringdato'].view('int64')

# Now fill NaNs in treatment_start
df['treatment_start'] = df['treatment_start'].fillna(df['diff_utleveringdato'])
print(df[['pasientlopenr', 'atckode', 'treatment_episode', 'treatment_start', 'treatment_end']].head())

df['treatment_start'].isna().sum()

# creating a new column as a readable datetime of 'treatment_start' and 'end'
df['treatment_start_2'] = pd.to_datetime(df['treatment_start'], unit='ns')
df['treatment_end_2'] = pd.to_datetime(df['treatment_end'], unit='ns')

df.columns

df = df.sort_values(by=['pasientlopenr', 'atckode'])

df.shape
df['diff_utleveringdato'] = pd.to_datetime(df['diff_utleveringdato'])
df[['atckode', 'diff_utleveringdato','treatment_start','treatment_end']].head(20)
# fill treatment start as diff_uteveind dato
df['treatment_start'] = df['diff_utleveringdato']
df[['atckode', 'diff_utleveringdato','treatment_start','treatment_end']].head(20)

# sum the duration of each ATC 
# Convert 'treatment_start_2' and 'treatment_end_2' to datetime
df['treatment_start'] = pd.to_datetime(df['treatment_start'])
df['treatment_end'] = pd.to_datetime(df['treatment_end'])

# Calculate the duration of each treatment episode in days
df['treatment_duration_days'] = (df['treatment_end'] - df['treatment_start']).dt.days

# Group the data by patient and ATC code to sum the durations
# Assuming 'pasient_fodselsar' uniquely identifies each patient (since no explicit patient ID is given)
summed_durations = df.groupby(['pasientlopenr', 'atckode'])['treatment_duration_days'].sum().reset_index()

summed_durations.head(10)
df.sort_values(by=['pasientlopenr', 'atckode'])

# Merge the summed durations back into the original dataset
df = pd.merge(df, summed_durations, on=['pasientlopenr', 'atckode'], suffixes=('', '_summed'))
df.head()
df.dtypes[['treatment_duration_days_summed','duration']]

# convert both columns to float
df['duration'] = df['duration'].astype(float)
df['percent_ATC_in_patient_duration'] = df['treatment_duration_days_summed']/df['duration']
df.to_csv(r'LMR_Ready_To_Merge.csv', index=False)
df['percent_ATC_in_patient_duration'] = df['treatment_duration_days_summed']/df['duration']


df[['percent_ATC_in_patient_duration','treatment_duration_days_summed','duration']].head()
df['percent_ATC_in_patient_duration'].describe()

# rename back the columns
df.rename(columns={
    'atckode': 'legemiddel_atckode_niva5',
    'pasientlopenr': 'lopenr',
    'diff_utleveringdato': 'utlevering_dato',
    'ordinasjonantallddd': 'utlevering_ddd'
}, inplace=True)

df.columns

df.to_csv(r'LMR_Ready_To_Merge.csv', index=False)

df['lopenr'].nunique()


df.shape

df['pasient_dodsarmaned'].isna().sum()

# Step 1: Filter the DataFrame where 'pasient_dodsarmaned' is not NaN
filtered_df = df[df['pasient_dodsarmaned'].notna()]

# Step 2: Get the unique values from 'lopenr'
unique_patients = filtered_df['lopenr'].nunique()

# Step 3: Output the number of unique patients
print(f"Number of unique patients with non-NaN 'pasient_dodsarmaned': {unique_patients}")
# next is to merge all "NPR_data_all.dta", "KPR_data_all_mapped_to_ICD.dta","LMR_data_all.dta



import pandas as pd
df= pd.read_stata(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.dta')

# Display max rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df.head()
# egenarte some indicator variables 
# 1. the patietn overall adherence, calculated as the avareg of his adherence over all drugs using PDC 
# if avarge adhrende is 80% or above adhered =1, if not not adhered = 0
df=pd.read_csv(r'LMR_data_all.csv')
df.head()


# making ttt episode 
# calculate the cohort duration of each patient in the dataset
# Convert 'dato' to datetime format
df['utlevering_dato'] = pd.to_datetime(df['utlevering_dato'])
# Groupby lopnr
patient_durations = df.groupby('lopenr')['utlevering_dato'].agg([min, max])

# Calculate the duration difference between the last and first prescription date
patient_durations['duration'] = (patient_durations['max'] - patient_durations['min']).dt.days

# Reset index to make 'lopenr' a column again
patient_durations.reset_index(inplace=True)
patient_durations.head()
# save 
df = pd.merge(df, patient_durations[['lopenr', 'duration']], on='lopenr', how='left')

# Rename some columns to match the code (should be renamed back later)
df.rename(columns={
    'legemiddel_atckode_niva5': 'atckode',
    'lopenr': 'pasientlopenr',
    'utlevering_dato': 'diff_utleveringdato',
    'utlevering_ddd': 'ordinasjonantallddd'
}, inplace=True)

df[['pasientlopenr', 'atckode', 'diff_utleveringdato', 'ordinasjonantallddd']].head(), df.dtypes[['pasientlopenr', 'atckode', 'diff_utleveringdato', 'ordinasjonantallddd']]
df['ordinasjonantallddd'] = df['ordinasjonantallddd'].str.replace(',', '.', regex=False)
df['utlevering_antallpakninger'] = df['utlevering_antallpakninger'].str.replace(',', '.', regex=False)
df['vare_ddd_verdi'] = df['vare_ddd_verdi'].str.replace(',', '.', regex=False)

# Convert to numeric, coercing errors to NaN
df['ordinasjonantallddd'] = pd.to_numeric(df['ordinasjonantallddd'], errors='coerce')
# step 1.Summation and Preparation:
# summing 'ordinasjonantallddd'
df['sum_ordinasjonantallddd'] = df.groupby(['pasientlopenr', 'atckode', 'diff_utleveringdato'])['ordinasjonantallddd'].transform('sum')

# Remove duplicates
df.drop_duplicates(subset=['pasientlopenr', 'atckode', 'diff_utleveringdato', 'ordinasjonantallddd'], inplace=True)

# sorting
df = df.sort_values(by=['pasientlopenr', 'atckode', 'diff_utleveringdato'])

# Lag and Lead Variables:
df['lag_dato'] = df.groupby(['pasientlopenr', 'atckode'])['diff_utleveringdato'].shift(1)
df['lead_dato'] = df.groupby(['pasientlopenr', 'atckode'])['diff_utleveringdato'].shift(-1)

# convert columns to datetime
df['lead_dato'] = pd.to_datetime(df['lead_dato'])
df['diff_utleveringdato'] = pd.to_datetime(df['diff_utleveringdato'])

# calculate the date differences
df['delta_dager'] = df['lead_dato'] - df['diff_utleveringdato']
df['delta_dager'] = df['delta_dager'].dt.days
df['delta_dager'].fillna(0, inplace=True),
df['ordinasjonantallddd'].head()

# Similar operations for DDD variables
df['lag_ddd'] = df.groupby(['pasientlopenr', 'atckode'])['ordinasjonantallddd'].shift(1)
df['lead_ddd'] = df.groupby(['pasientlopenr', 'atckode'])['ordinasjonantallddd'].shift(-1)

# Adherence calculation
df['ordinasjonantallddd'] = pd.to_numeric(df['ordinasjonantallddd'], errors='coerce')
df['ddd_80p_adh'] = df['ordinasjonantallddd'] / 0.8
df['lag_ddd_80p_adh'] = df.groupby(['pasientlopenr', 'atckode'])['ddd_80p_adh'].shift(1)
df['lead_ddd_80p_adh'] = df.groupby(['pasientlopenr', 'atckode'])['ddd_80p_adh'].shift(-1)
# Prescription coverage and carryover
df['resept_dekning_dager'] = df['ddd_80p_adh'] - df['delta_dager']
df['resept_dekning_dager'] = df['resept_dekning_dager'].apply(lambda x: max(x, 0))

df['lag_rx_dekn_dag'] = df.groupby(['pasientlopenr', 'atckode'])['resept_dekning_dager'].shift(1).fillna(0)

df['carryover'] = df['ddd_80p_adh'] - df['delta_dager'] + df['lag_rx_dekn_dag']
df['lag_carryover'] = df.groupby(['pasientlopenr', 'atckode'])['carryover'].shift(1)
df['treatment_episode'] = 0
df.loc[df['lag_carryover'].isna() | (df['lag_carryover'] < -14), 'treatment_episode'] = 1
df.loc[df['carryover'] < -14 | df['lead_dato'].isna(), 'treatment_episode'] = 3

# Set initial values for treatment start and end based on episodes
df['treatment_start'] = np.where(df['treatment_episode'] == 1, df['diff_utleveringdato'], pd.NaT)
df['treatment_end'] = np.where(df['treatment_episode'] == 3, df['lead_dato'], pd.NaT)

# Fill treatment start and end across all episodes
df['treatment_start'] = df.groupby(['pasientlopenr', 'atckode'])['treatment_start'].fillna(method='ffill')
df['treatment_end'] = df.groupby(['pasientlopenr', 'atckode'])['treatment_end'].fillna(method='bfill')

# Step 1: Summarize adherence per patient and per ATC code
# Calculate the average adherence for each patient and ATC combination
df['adherence'] = df['ddd_80p_adh'] / df['ordinasjonantallddd']

# Step 2: Calculate the average adherence per patient over all ATC codes
# Group by patient (lopenr) and calculate the mean adherence for each patient
patient_adherence = df.groupby('pasientlopenr')['adherence'].mean().reset_index()

# Step 3: Determine whether a patient is adhered or not
# If the average adherence is 0.80 or above, mark them as adhered (1), otherwise not adhered (0)
patient_adherence['adhered_status'] = np.where(patient_adherence['adherence'] >= 0.80, 1, 0)

# Step 4: Create the final DataFrame with unique patient id and adherence status
final_patient_adherence = patient_adherence[['pasientlopenr', 'adhered_status']]



final_patient_adherence['adhered_status'].value_counts()
final_patient_adherence.rename(columns={'pasientlopenr': 'lopenr'}, inplace=True)
final_patient_adherence.columns
final_patient_adherence.to_stata(r'adherence_status.dta')

# the file with merged using stata and saved here :Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.dta



# counting number of medication pr patient
df= pd.read_stata(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.dta')
df.head()
# Step 1: Count unique medications (ATC codes) per patient
unique_medications_per_patient = df.groupby('lopenr')['legemiddel_atckode_niva5'].nunique().reset_index()

# Rename the new column for clarity, keeping the original 'legemiddel_atckode_niva5' intact
unique_medications_per_patient.rename(columns={'legemiddel_atckode_niva5': 'medication_count_per_patient'}, inplace=True)

# Step 2: Merge this count back into the original dataframe
df = pd.merge(df, unique_medications_per_patient, on='lopenr', how='left')
df.head(10)


df.to_stata(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.dta')

df= pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all_reduced_size.csv')
df.head()
ddi_df = pd.read_csv(r'Mapping/DDIs_for_stacking_article.csv')
ddi_df.head()

# Step 1: Define the DDI filter
ddi_df_filtered = ddi_df[ddi_df['Grad'] == 3]

# Step 2: Reduce memory by downcasting numerical columns and changing types
df['utlevering_ddd'] = pd.to_numeric(df['utlevering_ddd'], downcast='float')
df['treatment_duration_days'] = pd.to_numeric(df['treatment_duration_days'], downcast='integer')
ddi_df_filtered['Grad'] = pd.to_numeric(ddi_df_filtered['Grad'], downcast='integer')

# Convert categorical/object columns (e.g., 'legemiddel_atckode_niva5') to category
df['legemiddel_atckode_niva5'] = df['legemiddel_atckode_niva5'].astype('category')
ddi_df_filtered['ATC1'] = ddi_df_filtered['ATC1'].astype('category')
ddi_df_filtered['ATC2'] = ddi_df_filtered['ATC2'].astype('category')

# Step 3: Convert 'utlevering_dato' column to datetime format
df['utlevering_dato'] = pd.to_datetime(df['utlevering_dato'], errors='coerce')

# Step 4: Use memory-efficient filtering instead of merging large datasets
atc1_set = set(ddi_df_filtered['ATC1'].unique())
atc2_set = set(ddi_df_filtered['ATC2'].unique())

# Filter the patient data to only include rows where the ATC code is in either ATC1 or ATC2
df_filtered_atc = df[df['legemiddel_atckode_niva5'].isin(atc1_set.union(atc2_set))]

# Step 5: Perform a self-join-like operation on the filtered data for each patient
interaction_flags = []

for patient_id, group in df_filtered_atc.groupby('lopenr'):
    atc_codes = group['legemiddel_atckode_niva5'].values
    atc_dates = group['utlevering_dato'].values

    found_interaction = False
    for i in range(len(atc_codes)):
        for j in range(i + 1, len(atc_codes)):
            if (atc_codes[i] in atc1_set and atc_codes[j] in atc2_set) or (atc_codes[i] in atc2_set and atc_codes[j] in atc1_set):
                # Check if the dates are within 30 days
                time_difference = np.abs((atc_dates[j] - atc_dates[i]).astype('timedelta64[D]')).astype(int)
                if time_difference <= 30:
                    found_interaction = True
                    break
        if found_interaction:
            break
    # Append the result (1 if interaction found, 0 otherwise)
    interaction_flags.append({'lopenr': patient_id, 'interaction_flag_3': int(found_interaction)})

# Step 6: Create a DataFrame from the interaction flags and merge it back into the original dataset
interaction_df = pd.DataFrame(interaction_flags)

# Merge the interaction flag back into the original dataframe
df = pd.merge(df, interaction_df, on='lopenr', how='left')

# Fill any missing interaction flags with 0
df['interaction_flag_3'].fillna(0, inplace=True)
df['interaction_flag_3'].value_counts()
df.head(10)
df.shape

df = df.rename(columns={'interaction_flag_3': 'interaction_flag'})
df.columns

df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all.csv', index= False)

df=pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all_reduced_size.csv')


columns = df.columns.tolist()
columns



# count medication wiht poteintial problems (from Drug Burden Index list) for each patient 
import pandas as pd

# Load the longitudinal dataset and the anticholinergic drug list
df = pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all_reduced_size.csv')
anticholinergic_drugs = pd.read_csv(r'Mapping/Drug_Burden_Index_DBI_List.csv')

# Extract the list of anticholinergic drugs from the column "ATC"
anticholinergic_list = anticholinergic_drugs['ATC'].tolist()

# Create a flag column to indicate if the drug in "legemiddel_atckode_niva5" is anticholinergic
df['is_anticholinergic'] = df['legemiddel_atckode_niva5'].isin(anticholinergic_list).astype(int)

# Group by 'lopenr' (patient ID) and count the number of anticholinergic drugs for each patient
anticholinergic_counts = df.groupby('lopenr')['is_anticholinergic'].sum().reset_index()

# Rename the column for clarity
anticholinergic_counts.rename(columns={'is_anticholinergic': 'DBI_drug_count'}, inplace=True)

# Merge the counts back into the original longitudinal dataframe
df = df.merge(anticholinergic_counts, on='lopenr', how='left')
df['DBI_drug_count'].hist()

df[['lopenr', 'DBI_drug_count']].head(50) #ok

# Save the updated dataframe if necessary
df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all_reduced_size.csv', index=False)

# Create a dictionary mapping the first character of the ATC code to medication categories
atc_categories = {
    'A': 'alimentary',
    'B': 'blood',
    'C': 'cardiovascular',
    'D': 'dermatologicals',
    'G': 'genitourinary',
    'H': 'sys hormonal',
    'J': 'anti infectives',
    'L': 'immunomodulating ',
    'M': 'musculoskeletal system',
    'N': 'nervous',
    'P': 'antiparasitic',
    'R': 'respiratory',
    'S': 'sensory',
    'V': 'various'
}

# Extract the first character of the ATC code to identify the category
df['atc_category'] = df['legemiddel_atckode_niva5'].str[0].map(atc_categories)

# Check for any unmatched categories (if any ATC code doesn't match, it will result in NaN)
unmatched_categories = df[df['atc_category'].isna()]
# Output the longitudinal_df to see the new column with medication categories
df.head()
df['atc_category'].hist()
# Save the updated dataframe if necessary
df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/LMR_data_all_reduced_size.csv', index=False)




