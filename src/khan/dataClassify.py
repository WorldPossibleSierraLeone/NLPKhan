import numpy as np
from dataTraining import model
from dataTraining import vectorizer
from targetEncoding import label_encoder
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import sys
import xlsxwriter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.textProcesser import *


os.makedirs('../../data_sets/khan_data/mapped_khan_data', exist_ok=True)

# Go through the final khan data set and make a prediction for each entry, and then add this column SL_Predicted_Unit_Name to each entry 
df = pd.read_csv('../../data_sets/khan_data/final_khan_data.csv')



# Preprocess the text data in the 'Unit_Name', 'Content_Name', and 'Unit_Sub_Unit_Name' columns
df['Unit_Name'] = df['Unit_Name'].apply(preprocess_text)
df['Content_Name'] = df['Content_Name'].apply(preprocess_text)
df['Unit_Sub_Unit_Name'] = df['Unit_Sub_Unit_Name'].apply(preprocess_text)

# Vectorize the preprocessed text data
X = vectorizer.transform(df['Unit_Name'] + ' ' + df['Content_Name'] + ' ' + df['Unit_Sub_Unit_Name'])

# Use the model to make predictions
predictions = model.predict(X)
predictions = label_encoder.inverse_transform(predictions)

# Add the predictions as a new column to the DataFrame
df['SL_Predicted_Unit_Name'] = predictions
random_df = df.sample(n=20)

# Now here comes the hard part: Let's create a seperate csv for each unique inversed target, and then add the rows that have that target to that csv. 
# But Order By: Level, then Unit Name, then Unit_Sub_Unit_Name, then Content_Name
unique_targets = np.unique(predictions)

writer = pd.ExcelWriter('../../data_sets/khan_data/mapped_khan_data/mapped_khan_data.xlsx', engine='xlsxwriter')

for target in unique_targets:
    target_df = df[df['SL_Predicted_Unit_Name'] == target]
    target_df = target_df.sort_values(by=['Level', 'Unit_Name', 'Unit_Sub_Unit_Name', 'Content_Name'])
    target_df.to_excel(writer, sheet_name=str(target), index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.close()