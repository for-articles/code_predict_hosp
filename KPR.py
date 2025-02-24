
# In STATA:
#===================
# We merged m:m all KPR files on 'lopenr' and 'regning_nokkel', the file is called "KPR_data_All.dta' (done)
# 101,634 didn't match from 'KPR_takstkode_1948'. By inspecting, they are tannlege eller fysio visits --> removed (done)
# too little rows for ICPC-2B and ICD-DA-3 and BUP-Akse (254 rows) -->removed (done)
# wrongly registered codes (A27, and S422) were corrected --> Done
# wrongly registered codes under ICPC classification were removed (from -30 to -69 code) --> Done
# We mapped ICPC to ICD-10 using 'Map_ICPC_To_ICD_2024_Small.dta' and is saved to file 'Files_After_Manipulation/KPR_data_all_mapped_to_ICD.dta'


import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_stata(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all.dta')
df.head()


# make varaibles of previous visits (1, 3, 6 months and one year)
# Step 1: Convert 'index_date' and 'dato_KPR' to datetime format
df.loc[:, 'index_date'] = pd.to_datetime(df['index_date'])
df.loc[:, 'dato_KPR'] = pd.to_datetime(df['dato_KPR'])

# Step 2: Drop duplicates by 'lopenr', 'regning_nokkel_KPR', and 'dato_KPR' to avoid counting duplicate visits
df_unique_visits = df.drop_duplicates(subset=['lopenr', 'regning_nokkel_KPR', 'dato_KPR']).copy()

# Step 3: Define time ranges (1, 3, 6 months, and 1 year before the index_date)
df_unique_visits.loc[:, 'one_month_before'] = df_unique_visits['index_date'] - pd.DateOffset(months=1)
df_unique_visits.loc[:, 'three_months_before'] = df_unique_visits['index_date'] - pd.DateOffset(months=3)
df_unique_visits.loc[:, 'six_months_before'] = df_unique_visits['index_date'] - pd.DateOffset(months=6)
df_unique_visits.loc[:, 'one_year_before'] = df_unique_visits['index_date'] - pd.DateOffset(months=12)

# Step 4: Create indicators for each period by checking if 'dato_KPR' falls within the respective ranges
df_unique_visits.loc[:, 'one_month_visits'] = df_unique_visits.apply(
    lambda x: 1 if x['one_month_before'] <= x['dato_KPR'] <= x['index_date'] else 0, axis=1)
df_unique_visits.loc[:, 'three_months_visits'] = df_unique_visits.apply(
    lambda x: 1 if x['three_months_before'] <= x['dato_KPR'] <= x['index_date'] else 0, axis=1)
df_unique_visits.loc[:, 'six_months_visits'] = df_unique_visits.apply(
    lambda x: 1 if x['six_months_before'] <= x['dato_KPR'] <= x['index_date'] else 0, axis=1)
df_unique_visits.loc[:, 'one_year_visits'] = df_unique_visits.apply(
    lambda x: 1 if x['one_year_before'] <= x['dato_KPR'] <= x['index_date'] else 0, axis=1)

# Step 5: Count total distinct visits (by regning_nokkel_KPR) per patient regardless of time
df_unique_visits['total_visits'] = df_unique_visits.groupby('lopenr')['regning_nokkel_KPR'].transform('nunique')

# Step 6: Group by 'lopenr' and sum up the visits in each time period
visit_counts = df_unique_visits.groupby('lopenr').agg(
    one_month_visits=('one_month_visits', 'sum'),
    three_months_visits=('three_months_visits', 'sum'),
    six_months_visits=('six_months_visits', 'sum'),
    one_year_visits=('one_year_visits', 'sum'),
    total_visits=('total_visits', 'max')  # Use 'max' since total_visits is the same for each group
).reset_index()

# Step 7: Merge the total visit counts from 'visit_counts' back into 'df_unique_visits'
df_unique_visits = df_unique_visits.merge(visit_counts, on='lopenr', suffixes=('', '_total'))

# Step 8: Save the updated 'df_unique_visits' DataFrame with visit counts to a CSV
df_unique_visits.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all.csv', index=False)
df_unique_visits.head()


df_unique_visits.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all.csv')



# count how many fastlege, kiropraktor, fysioterap, ... for each patient 
import pandas as pd
# Create a DataFrame
df = pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all.csv')

# Convert 'index_date' and 'dato_KPR' to datetime format
df.loc[:, 'index_date'] = pd.to_datetime(df['index_date'])
df.loc[:, 'dato_KPR'] = pd.to_datetime(df['dato_KPR'])

# Drop duplicates by 'lopenr', 'regning_nokkel_KPR', and 'dato_KPR' to avoid counting duplicate visits
df_unique_visits_type = df.drop_duplicates(subset=['lopenr', 'regning_nokkel_KPR', 'dato_KPR']).copy()

# Count unique visits (by regning_nokkel_KPR) for each patient ('lopenr') and visit type ('tjenestetype_KPR')
visit_type_counts = df_unique_visits_type.groupby(['lopenr', 'tjenestetype_KPR']).agg(
    unique_visits=('regning_nokkel_KPR', 'nunique')
).reset_index()

# Reshape the data to have one column for each visit type
visit_type_pivot = visit_type_counts.pivot(index='lopenr', columns='tjenestetype_KPR', values='unique_visits').fillna(0)

# Flatten the MultiIndex in the column headers (optional, for cleaner output)
visit_type_pivot.columns.name = None
visit_type_pivot = visit_type_pivot.reset_index()

# Merge the visit type counts back to the original df_unique_visits DataFrame
df_merged = df_unique_visits_type.merge(visit_type_pivot, on='lopenr', how='left')

# Display the result
print(df_merged)
df_merged.columns

df_merged['tannlege'].nunique()

df_merged.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all.csv')

df= pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all.csv')

# Display max rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df.head()


import pandas as pd
df= pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv')
df.shape
df.head()

# genarte some indiacator variables 
# 1. Trend in visits frequencey before index date
def detect_logical_trend(row):
    visits = {
        'one_month': row['one_month_visits'],
        'three_months': row['three_months_visits'],
        'six_months': row['six_months_visits'],
        'one_year': row['one_year_visits']
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
df_unique_patients = df[['lopenr', 'one_month_visits', 'three_months_visits', 'six_months_visits', 'one_year_visits']].drop_duplicates()

# Apply the logic to detect overall visit trend for each unique patient
df_unique_patients['visit_trend'] = df_unique_patients.apply(detect_logical_trend, axis=1)
df_unique_patients = df_unique_patients.drop_duplicates(subset=['lopenr'])

df_unique_patients['visit_trend'].hist()


# Now merge this trend back to the original DataFrame on 'lopenr' to ensure each row for the same patient gets the trend value
df = df.merge(df_unique_patients[['lopenr', 'visit_trend']], on='lopenr', how='left')
df.head()

df.drop(['visit_trend_x'], axis=1 , inplace= True)

df.rename(columns={'visit_trend_y': 'visit_trend'}, inplace= True)
df.head()

df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv', index=False)

# 2. Time since last GP visit
# Convert 'index_date' and 'dato_kpr' columns to datetime
df['index_date'] = pd.to_datetime(df['index_date'])
df['dato_kpr'] = pd.to_datetime(df['dato_kpr'])

# Group by 'lopenr' to calculate the most recent GP visit per patient
def calculate_days_since_last_gp_visit(df_group):
    # Filter for GP visits within the group (where 'tjenestetype_kpr' is 'FL')
    gp_visits = df_group[df_group['tjenestetype_kpr'] == 'FL']
    
    if not gp_visits.empty:
        # Get the most recent GP visit date for this patient
        most_recent_gp_visit = gp_visits['dato_kpr'].max()
        # Calculate the days since the last GP visit
        days_since_last_gp = (df_group['index_date'].iloc[0] - most_recent_gp_visit).days
    else:
        # If no GP visit, assign -1
        days_since_last_gp = -1
    
    # Assign the calculated value to all rows for this patient
    df_group['days_since_last_gp_visit'] = days_since_last_gp
    return df_group

# Apply the function to each patient group
df = df.groupby('lopenr', group_keys=False).apply(calculate_days_since_last_gp_visit)
df.head()
df.shape

df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv', index=False)

# 3. make a varaible that indicates subtance use (ICD-10 codes F10-F19)
# Create a new variable 'substance_use' indicating whether the patient has an ICD-10 code between F10 and F19
df['substance_use'] = df['icd10kode_kpr'].apply(lambda x: 1 if x.startswith('F1') else 0)

# Ensure that the flag is consistent across all rows for the same patient by grouping by 'lopenr'
df['substance_use'] = df.groupby('lopenr')['substance_use'].transform('max')
df['substance_use'].value_counts()

df.head()
df.to_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv', index=False)

# next step is to merge with npr
df= pd.read_csv(r'Files_After_Manipulation/After_removing_history_before_index_date/KPR_data_all_reduced_size.csv')
columns = df.columns.tolist()
columns
# counting lab tests has been done in stata, see stata.do file

