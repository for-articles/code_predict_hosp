
# STUDY DESIGN
#===================
# define the intervention =1 and the controls=0 (done)
# history starts 01.01.2008 and until 01.01.2018 for all patients (interventiona nd controls) (ok)
# the follow up period is 2018-2019 (ok)
# the end point (index date) of the intervension group is the first admission happens in this period 2018-2019 for each admitted patient (ok)
# to ensure equality we measure the distribution of the intervention group endpoit dates and simulate the same distribution for the contols (ok)
# history which comes after the endpoint removed 
# non-elective admissions are maintained and others are removed (ok)
# LMR patients that are not in the NPR files are removed 
# filter out the variabels you won't use in the model
# keep history od diagnoses (from 2008) 


# 1. Define study population
#============================
import pandas as pd
import numpy as np

# Display max rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_stata(r'archive/Masterfiler for Mohsen/aktivitet_1_1948.dta')
df.columns
df.head(10)

import glob
import os

# Define the directory and pattern to match all NPR files
directory = r'archive/Masterfiler for Mohsen/'
file_pattern = os.path.join(directory, "aktivitet_*.dta")

# Get a list of all matching files
files = glob.glob(file_pattern)

# Initialize an empty series to store all unique innmatehast counts
innmatehast_counts = pd.Series(dtype='int')

# Loop through each file
for file in files:
    # Load the Stata (.dta) file into a pandas DataFrame
    df = pd.read_stata(file)
    
    # Ensure 'innmatehast' is treated as numeric, removing any leading/trailing spaces
    if 'innmatehast' in df.columns:
        df['innmatehast'] = pd.to_numeric(df['innmatehast'], errors='coerce')
    
    # Check if 'utdato' and 'inndato' columns exist
    if 'utdato' in df.columns and 'inndato' in df.columns:
        # Convert 'utdato' and 'inndato' to datetime
        df['utdato'] = pd.to_datetime(df['utdato'], errors='coerce')
        df['inndato'] = pd.to_datetime(df['inndato'], errors='coerce')
        
        # Create a new 'year' column based on 'inndato'
        df['year'] = df['inndato'].dt.year # we use 'year' instead of 'aar because 'aar' isn't accurate compare to 'inndato'
        
        # Compute the length of stay in days (using the datetime objects)
        df['los'] = (df['utdato'] - df['inndato']).dt.days
        
        # Ensure 'innmatehast', 'year', and 'lopenr' columns exist
        if 'innmatehast' in df.columns and 'year' in df.columns and 'los' in df.columns and 'lopenr' in df.columns:
            # Apply the filter: innmatehast != 4, year >= 2018, and los > 0
            filtered_df = df[(df['innmatehast'] != 4) & (df['year'] >= 2018) & (df['los'] > 0)]
            
            # Count the unique 'lopenr' values
            counts = filtered_df['lopenr'].value_counts()
            
            # Add the counts to the running total
            innmatehast_counts = innmatehast_counts.add(counts, fill_value=0)

# Convert the series to a DataFrame
intervention_df = innmatehast_counts.reset_index()
intervention_df.columns = ['lopenr', 'count']

# Display the result
print(intervention_df)

intervention_df['lopenr'].nunique() #9379 

# save intervention_df
intervention_df.to_stata('Files_After_Manipulation/intervention_group_lopenr.dta', write_index=False)


# define patiets as intervention=(1) or control=(0) by merging with (intervention_df)
# Ensure the 'lopenr' column exists in both DataFrames
lopenr_1948_df = pd.read_stata(r'archive/Masterfiler for Mohsen/lopenr_1948.dta')

if 'lopenr' in lopenr_1948_df.columns and 'lopenr' in intervention_df.columns:
    
    # Create a 'group' column in lopenr_1948_df where 'lopenr' is in intervention_df
    lopenr_1948_df['group'] = lopenr_1948_df['lopenr'].isin(intervention_df['lopenr']).astype(int)

# Save the result to a new CSV file
output_file = 'Files_After_Manipulation/lopenr_1948_with_group.dta'
lopenr_1948_df.to_stata(output_file, write_index=False)

# Optional: Display the result
print(f"Updated DataFrame saved to {output_file}")


df_key =pd.read_stata(r'Files_After_Manipulation/lopenr_1948_with_group.dta')
df_key['group'].value_counts() #58313 controls, 9414 interventions (total 67727)
df_key['lopenr'].tail(5)
# IN STATA
#==============
# we merged NPR with diagnoses file
# we merged in Stata (m:m) on lopenr and episode_nokkel and saved to 'NPR_data_all.dta'
# we merged with (m:1) 'group' from lopenr_1948_with_group on 'lopenr' 
# The final file called "Files_After_Manipulation/NPR_data_all.dta"
# We determined the index date for the intervention group (=1), file 'Files_After_Manipulation/intervention_with_index_date.dta'
# We made a file for unique 'lopenr' control group (=0), file 'Files_After_Manipulation/control_with_index_date.dta'

#------------------------------------------------------------------------------------
# 2. Simulate the disteribution of the intervention group index date to the control group 
#-------------------------------------------------------------------------------------
# 1. plot distribution of the intervention group
import matplotlib.pyplot as plt

# Load the intervention group data
intervention_df = pd.read_stata(r'Files_After_Manipulation/intervention_with_index_date.dta')

# Ensure 'index_date' is treated as a datetime object
intervention_df['index_date'] = pd.to_datetime(intervention_df['index_date'], errors='coerce')

# Plot the distribution of index dates for the intervention group
plt.figure(figsize=(10, 6))
intervention_df['index_date'].hist(bins=30, edgecolor='black', alpha=0.5, label='Intervention group', color='orange')
plt.title('Distribution of index dates in the intervention group')
plt.xlabel('Index Date')
plt.ylabel('Frequency')
plt.legend()
plt.grid(False)
plt.show()

# Load the control group data
control_df = pd.read_stata(r'Files_After_Manipulation/control_with_index_date.dta')

# 2. Simulate index dates for the control group by sampling from the intervention group
control_df['simulated_index_date'] = intervention_df['index_date'].sample(n=len(control_df), replace=True).reset_index(drop=True)

# 3. Plot the distribution of simulated index dates for the control group
plt.figure(figsize=(10, 6))
control_df['simulated_index_date'].hist(bins=30, edgecolor='black', alpha=0.5, label='Control Group (Simulated)', color='green')
plt.title('Simulated distribution of index dates for control Group')
plt.xlabel('Simulated Index Date')
plt.ylabel('Frequency')
plt.legend()
plt.grid(False)
plt.show()

# Rename 'simulated_index_date' to 'index_date'
control_df = control_df.rename(columns={'simulated_index_date': 'index_date'})

# 4. Save the updated control_df with the simulated index dates back to a Stata file
output_file = r'Files_After_Manipulation/control_with_index_date.csv'  
control_df.to_csv(output_file, index=False)
print(f"Simulated index dates saved to {output_file}")

# 5. Plot both distributions: intervention vs. simulated control
plt.figure(figsize=(10, 6))
# Define ggplot2-like default colors
ggplot2_colors = ['#F8766D', '#00BFC4']
# Plot intervention group
intervention_df['index_date'].hist(bins=30, edgecolor='black', alpha=0.5, label='Intervention Group', color='orange')

# Plot simulated control group
control_df['simulated_index_date'].hist(bins=30, edgecolor='black', alpha=0.3, label='Control Group (Simulated)', color='green')

plt.title('Comparison of index date distributions')
plt.xlabel('Index Date')
plt.ylabel('Frequency')
plt.legend()
plt.grid(False)
plt.show()

# plot both index_date of intervention group and simualted_index_date for the control group
import pandas as pd
import matplotlib.pyplot as plt

# Load the intervention group data
intervention_df = pd.read_stata(r'Files_After_Manipulation/intervention_with_index_date.dta')

# Ensure 'index_date' is treated as a datetime object
intervention_df['index_date'] = pd.to_datetime(intervention_df['index_date'], errors='coerce')

# Load the control group data
control_df = pd.read_stata(r'Files_After_Manipulation/control_with_index_date.dta')

# Simulate index dates for the control group by sampling from the intervention group
control_df['simulated_index_date'] = intervention_df['index_date'].sample(n=len(control_df), replace=True).reset_index(drop=True)

# Group dates by month (or choose 'D' for day, 'W' for week)
intervention_df['index_date_binned'] = intervention_df['index_date'].dt.to_period('M')  # You can change 'M' to 'W' for week
control_df['simulated_index_date_binned'] = control_df['simulated_index_date'].dt.to_period('M')

# Count the number of occurrences in each bin
intervention_counts = intervention_df['index_date_binned'].value_counts().sort_index()
control_counts = control_df['simulated_index_date_binned'].value_counts().sort_index()

# Plot the line plot for both groups
plt.figure(figsize=(10, 6))

# Line plot for the intervention group
plt.plot(intervention_counts.index.astype(str), intervention_counts.values, label='Intervention Group', color='orange', lw=2)

# Line plot for the control group
plt.plot(control_counts.index.astype(str), control_counts.values, label='Control Group (Simulated)', color='blue', lw=2)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Set plot labels and title
plt.title('')
plt.xlabel('Index Date')
plt.ylabel('Patients count')

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
# after saving control_with_index_date.csv was imported to stata and saved as control_with_index_date.dta
# the file was appended with intervention_with_index_date.dta and saved a one key filr for tye whole population 
# the file 'lopenr_1948_with_group_index_date.dta' conatins 'lopene','group', and 'index_date' columns
key_df_all= pd.read_stata(r'Files_After_Manipulation/study_population.dta')
key_df_all.head()
key_df_all['group'].value_counts()





