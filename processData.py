import pandas as pd
import numpy as np

khan_academy_data = pd.read_csv('data_sets/final_khan_data.csv')

# Drop the 'Unit_Sub_Unit_url' column
khan_academy_data = khan_academy_data.drop('Unit_Sub_Unit_url', axis=1)

# Move the 'Unit_Description' column to the end
cols = list(khan_academy_data.columns)
cols.append(cols.pop(cols.index('Unit_Description')))
khan_academy_data = khan_academy_data[cols]

# Calculate 2.5% of the total number of rows
sample_size = int(0.025 * len(khan_academy_data))

# Round to the nearest multiple of 10
sample_size = round(sample_size / 10) * 10

# Randomly select rows
khan_train_data = khan_academy_data.sample(n=sample_size, random_state=42)

# Write the sampled data to a new CSV file
khan_train_data.to_csv('data_sets/khan_train_data.csv', index=False)