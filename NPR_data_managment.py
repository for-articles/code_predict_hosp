
# Append all NPR files (Done)
# merge with diagnoses (Done)

# loop through LMR files
# merge KPR files
# make ttt episodes 
# filter out unnecssary diseases (non-chronic?)

import pandas as pd
# append all NPR files in one file 
import glob
import os

# Define the directory and pattern to match all .dta files
directory = r'archive/Masterfiler for Mohsen/'
file_pattern = os.path.join(directory, "aktivitet_*.dta")

# Get a list of all matching files
files = glob.glob(file_pattern)

# Initialize an empty list to hold DataFrames
df_list = []

# Loop through each file and append it to the list
for file in files:
    # Load the Stata (.dta) file into a pandas DataFrame
    df = pd.read_stata(file)
        # Check for problematic columns and convert them to strings, handling null values
    if 'aktivitetskategori3' in df.columns:
        # Convert the column to string, replacing any problematic or null values
        df['aktivitetskategori3'] = df['aktivitetskategori3'].astype(str).replace({pd.NA: None})
    if 'liggetid' in df.columns:
        # Try converting 'liggetid' to numeric, force errors to NaN
        df['liggetid'] = pd.to_numeric(df['liggetid'], errors='coerce')
    if 'uttilstand' in df.columns:
        # Try converting 'uttilstand' to numeric, force errors to NaN
        df['uttilstand'] = pd.to_numeric(df['uttilstand'], errors='coerce')
    
    # Append the DataFrame to the list
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame to a new file (e.g., CSV or Stata file)
output_file = 'NPR_data_all_no_group.dta'  # You can save as .dta too by changing to .dta extension
combined_df.to_stata(output_file, write_index=False)

# Optional: Print a message to confirm
print(f"All files have been combined and saved to {output_file}")

# IN STATA
#==============
# merge NPR with diagnoses file (done)
# merged in Stata (m:m) on lopenr and episode_nokkel and saved to 'NPR_data_all.dta' (done)
# merge with (m:1) 'group' from lopenr_1948_with_group on 'lopenr' (done) (done)
# final file called "Files_After_Manipulation/NPR_data_all.dta" (done)
# determine the index date for the intervention group (=1), file 'Files_After_Manipulation/intervention_with_index_date.dta' (done)
# make a file for unique 'lopenr' control group (=0), file 'Files_After_Manipulation/control_with_index_date.dta' (done)

import pandas as pd
df = pd.read_csv(
    r'Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all.csv', 
    encoding="ISO-8859-1",
    dtype={15: str, 34: str, 35: str}
)

df.columns

# Display max rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df.head()
df.columns

# generate some indicator varaibles
# LOS
#-------
# Modify the 'inndato' and 'utdato' conversion with error handling
df['inndato'] = pd.to_datetime(df['inndato'], errors='coerce')
df['utdato'] = pd.to_datetime(df['utdato'], errors='coerce')

# Recalculate Length of Stay where valid dates are present
df['lengthofstay'] = (df['utdato'] - df['inndato']).dt.days


# 2. admission seasonality:
#----------------------------
# Define a function to determine the season based on the month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Extract month from 'inndato' and determine the season
df['admission_season'] = df['inndato'].dt.month.apply(get_season)


# 3. number fo admission prior to index_Date
df['index_date'] = pd.to_datetime(df['index_date'], errors='coerce')

# Now we will create a column for previous admissions by counting admissions before the index_date for each patient (lopenr)
df = df.sort_values(['lopenr', 'inndato'])  # Sort by patient and admission date to ensure proper chronological order

def count_total_previous_admissions(patient_data):
    # Count all admissions that happened before the index_date for this patient
    total_previous = (patient_data['inndato'] < patient_data['index_date']).sum()
    # Assign this count to all rows for this patient
    patient_data['previous_admissions'] = total_previous
    return patient_data

# Apply the function to each patient group
df = df.groupby('lopenr').apply(count_total_previous_admissions)

# GENERATE NPR FEATURES
###########################

import pandas as pd

df = pd.read_csv(
    r'\NPR_data_all_reduced_size.csv', 
    dtype={15: str, 31: str, 32: str}, 
    delimiter=","
    )

# Display max rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# generate some indicator varaibles
# 1. LOS
#-------
# present in 'liggetid' varaible

df.head(73)

# 2. admission seasonality:
#----------------------------
# Define a function to determine the season based on the month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Extract month from 'inndato' and determine the season
df['admission_season'] = df['inndato'].dt.month.apply(get_season)

# 3. number of admission prior to index_Date
#-----------------------------------------------


# Define a function to calculate time since the last admission for each patient
# Sort the data by patient and admission date
df = df.sort_values(['lopenr', 'inndato'])

# Define the lambda function to calculate the maximum previous admissions
df['previous_admissions'] = df.groupby('lopenr').apply(
    lambda group: (group['inndato'] < group['index_date']).cumsum() - 1
).groupby('lopenr').transform('max').reset_index(drop=True)


# save 
df.to_csv(r'\NPR_data_all.csv', index=False)

# 4. Time since last admission before index date
#-----------------------------------------------
# Efficiently calculate time since last admission using shift within each patient group
# Sort by patient and admission date to ensure proper order
df = df.sort_values(['lopenr', 'inndato'])

# Convert both 'inndato' and 'index_date' to datetime to ensure compatibility
df['inndato'] = pd.to_datetime(df['inndato'], errors='coerce')
df['index_date'] = pd.to_datetime(df['index_date'], errors='coerce')

# Step 1: Find the most recent admission before the index date for each patient
most_recent_admission = df[df['inndato'] < df['index_date']].groupby('lopenr')['inndato'].max()

# Step 2: Merge this back into the original dataframe, so we can calculate the difference
df = df.merge(most_recent_admission.rename('last_admission_date'), on='lopenr', how='left')

# Step 3: Calculate the time since last admission in days
df['time_since_last_admission'] = (df['index_date'] - df['last_admission_date']).dt.days

# Step 4: If there's no previous admission before the index date, set the value to NaN
df['time_since_last_admission'] = df['time_since_last_admission'].where(df['last_admission_date'].notna(), None)

# Drop the 'last_admission_date' column if it's no longer needed
df.drop(columns=['last_admission_date'], inplace=True)

# save 
df.to_csv(r'\NPR_data_all.csv', index=False)

# 5. calcualte Charlson Comorbidity Index
#-----------------------------------------------

# Step 1: Define a comprehensive ICD-10 to Charlson Comorbidity mapping
charlson_conditions = {
    'Myocardial Infarction': {'codes': ['I21', 'I22', 'I25.2'], 'weight': 1},
    'Congestive Heart Failure': {'codes': ['I50', 'I11.0'], 'weight': 1},
    'Peripheral Vascular Disease': {'codes': ['I70', 'I71'], 'weight': 1},
    'Cerebrovascular Disease': {'codes': ['I60', 'I61', 'I62', 'G45'], 'weight': 1},
    'Dementia': {'codes': ['F00', 'F01', 'F02', 'F03'], 'weight': 1},
    'Chronic Pulmonary Disease': {'codes': ['J40', 'J41', 'J42', 'J43', 'J44', 'J45'], 'weight': 1},
    'Rheumatic Disease': {'codes': ['M05', 'M06', 'M32', 'M34'], 'weight': 1},
    'Peptic Ulcer Disease': {'codes': ['K25', 'K26', 'K27', 'K28'], 'weight': 1},
    'Mild Liver Disease': {'codes': ['K70.0', 'K70.3', 'K73', 'K74'], 'weight': 1},
    'Diabetes without End Organ Damage': {'codes': ['E10', 'E11'], 'weight': 1},
    'Diabetes with End Organ Damage': {'codes': ['E10.7', 'E11.7'], 'weight': 2},
    'Hemiplegia or Paraplegia': {'codes': ['G81', 'G82'], 'weight': 2},
    'Renal Disease': {'codes': ['N18', 'N19'], 'weight': 2},
    'Any Malignancy': {'codes': ['C00', 'C97'], 'weight': 2},
    'Metastatic Solid Tumor': {'codes': ['C77', 'C78', 'C79', 'C80'], 'weight': 6},
    'Severe Liver Disease': {'codes': ['K72', 'K76.6'], 'weight': 3},
    'AIDS/HIV': {'codes': ['B20', 'B22', 'B24'], 'weight': 6}
}

# Step 1: Reverse mapping from codes to conditions for faster lookup
icd_to_condition = {}
for condition, details in charlson_conditions.items():
    for code in details['codes']:
        icd_to_condition[code] = {'condition': condition, 'weight': details['weight']}

# Step 2: Convert 'kodeverdi' to string (if necessary)
df['kodeverdi'] = df['kodeverdi'].astype(str)

# Step 3: Function to map diagnosis codes to Charlson comorbidities
def calculate_charlson_comorbidity_index(diagnosis_codes):
    comorbidities_found = set()
    total_score = 0
    
    for code in diagnosis_codes:
        if pd.notna(code):
            code = str(code)
            # Use reverse lookup to find matching conditions
            for icd_prefix in icd_to_condition:
                if code.startswith(icd_prefix):
                    condition = icd_to_condition[icd_prefix]['condition']
                    weight = icd_to_condition[icd_prefix]['weight']
                    if condition not in comorbidities_found:
                        comorbidities_found.add(condition)
                        total_score += weight
                    break  # Stop checking once the first match is found (to optimize time)
    
    return total_score

# Step 4: Filter rows where 'kodenavn' indicates the code is ICD-10
df_icd = df[df['kodenavn'] == 'ICD-10']

# Step 5: Group by 'lopenr' and calculate Charlson Comorbidity Index for each patient
# Instead of using transform, apply the function directly
charlson_scores = df_icd.groupby('lopenr')['kodeverdi'].apply(calculate_charlson_comorbidity_index).reset_index()

# Rename the column to 'charlson_comorbidity_index' for clarity
charlson_scores.rename(columns={'kodeverdi': 'charlson_comorbidity_index'}, inplace=True)

# Step 6: Merge the result back to the original df
df = df.merge(charlson_scores, on='lopenr', how='left', suffixes=('', '_charlson'))

# Step 7: Fill missing CCI scores with 0 for patients with no ICD-10 codes
df['charlson_comorbidity_index'] = df['charlson_comorbidity_index'].fillna(0)

# Step 8: Deduplicate if needed (to keep only unique patients with their CCI score)
df_unique = df.drop_duplicates(subset=['lopenr'], keep='first').reset_index(drop=True)

# Now `df_unique` contains the Charlson Comorbidity Index for each patient
print(df_unique[['lopenr', 'charlson_comorbidity_index']]) # ok

df.to_csv(r'\NPR_data_all_reduced_size.csv', index=False)

import pandas as pd
df = pd.read_csv(
    r'Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all_reduced_size.csv', 
    dtype={15: str, 29: str, 48: str}
    )

columns = df.columns.tolist()
columns

# make some useful variables for the prediction
# number of visits 1, 3, 6, and one year prior to index date 

# Convert 'index_date' and 'dato_KPR' to datetime format
df.loc[:, 'index_date'] = pd.to_datetime(df['index_date'])
df.loc[:, 'inndato'] = pd.to_datetime(df['inndato'])

# Drop duplicates by 'lopenr', 'regning_nokkel_KPR', and 'dato_KPR' to avoid counting duplicate visits
df_unique_admissions = df.drop_duplicates(subset=['lopenr', 'episode_nokkel', 'inndato']).copy()

# Define time ranges (1, 3, 6 months, and 1 year before the index_date)
df_unique_admissions.loc[:, 'adm_one_month_before'] = df_unique_admissions['index_date'] - pd.DateOffset(months=1)
df_unique_admissions.loc[:, 'adm_three_months_before'] = df_unique_admissions['index_date'] - pd.DateOffset(months=3)
df_unique_admissions.loc[:, 'adm_six_months_before'] = df_unique_admissions['index_date'] - pd.DateOffset(months=6)
df_unique_admissions.loc[:, 'adm_one_year_before'] = df_unique_admissions['index_date'] - pd.DateOffset(months=12)

# Create indicators for each period by checking if 'dato_KPR' falls within the respective ranges
df_unique_admissions.loc[:, 'one_month_admissions'] = df_unique_admissions.apply(
    lambda x: 1 if x['adm_one_month_before'] <= x['inndato'] <= x['index_date'] else 0, axis=1)
df_unique_admissions.loc[:, 'three_months_admissions'] = df_unique_admissions.apply(
    lambda x: 1 if x['adm_three_months_before'] <= x['inndato'] <= x['index_date'] else 0, axis=1)
df_unique_admissions.loc[:, 'six_months_admissions'] = df_unique_admissions.apply(
    lambda x: 1 if x['adm_six_months_before'] <= x['inndato'] <= x['index_date'] else 0, axis=1)
df_unique_admissions.loc[:, 'one_year_admissions'] = df_unique_admissions.apply(
    lambda x: 1 if x['adm_one_year_before'] <= x['inndato'] <= x['index_date'] else 0, axis=1)

# Count total distinct visits (by regning_nokkel_KPR) per patient regardless of time
df_unique_admissions['total_admissions'] = df_unique_admissions.groupby('lopenr')['episode_nokkel'].transform('nunique')

# Group by 'lopenr' and sum up the visits in each time period
admissions_counts = df_unique_admissions.groupby('lopenr').agg(
    one_month_admissions=('one_month_admissions', 'sum'),
    three_months_admissions=('three_months_admissions', 'sum'),
    six_months_admissions=('six_months_admissions', 'sum'),
    one_year_admissionss=('one_year_admissions', 'sum'),
    total_admissions=('total_admissions', 'max')  # Use 'max' since total_admissions is the same for each group
).reset_index()

# Display the result
print(admissions_counts)

df = pd.merge(df, admissions_counts, on='lopenr', how='left')
df.sort_values(by=['lopenr'])
df.head()

df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all_reduced_size.csv', index = False)

import pandas as pd
df = pd.read_csv(
    r'Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all_reduced_size.csv', 
    dtype={15: str, 29: str, 48: str}
    )

df.rename(columns={'one_year_admissionss': 'one_year_admissions'}, inplace=True)

# genarte some indiacator variables 
# 1. Trend in visits frequencey before index date
def detect_logical_trend(row):
    visits = {
        'one_month': row['one_month_admissions'],
        'three_months': row['three_months_admissions'],
        'six_months': row['six_months_admissions'],
        'one_year': row['one_year_admissions']
    }
    
    # Compare visits over decreasing time windows
    if visits['one_month']/1 >= visits['three_months']/3 >= visits['six_months']/6 >= visits['one_year']/12:
        return "i"
    elif visits['one_month']/1 >= visits['three_months']/3 >= visits['six_months']/6 >= visits['one_year']/12:
        return "d"
    elif visits['one_month']/1 >= visits['three_months']/3 >= visits['six_months']/6 >= visits['one_year']/12:
        return "s"
    else:
        return "f"

# Create a new DataFrame by selecting the relevant columns and dropping duplicates based on 'lopenr'
df_unique_patients = df[['lopenr', 'one_month_admissions', 'three_months_admissions', 'six_months_admissions', 'one_year_admissions']].drop_duplicates()

# Apply the logic to detect overall visit trend for each unique patient
df_unique_patients['admissions_trend'] = df_unique_patients.apply(detect_logical_trend, axis=1)
df_unique_patients = df_unique_patients.drop_duplicates(subset=['lopenr'])
df_unique_patients.head()

# Now merge this trend back to the original DataFrame on 'lopenr' to ensure each row for the same patient gets the trend value
df = df.merge(df_unique_patients[['lopenr', 'admissions_trend']], on='lopenr', how='left')


df['admissions_trend'].hist()

check = pd.crosstab(df['group'], df['admissions_trend'])
check
df.head()
df.shape

df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all_reduced_size.csv', index = False)

df= pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/NPR_data_all_reduced_size.csv')
df.columns.tolist()
