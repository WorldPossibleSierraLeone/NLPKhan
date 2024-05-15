import numpy as np
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
import string
import nltk
import ssl
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.textProcesser import *
from targetEncoding import *

df = pd.read_csv('../../data_sets/khan_data/khan_train_data.csv')


# Now lets build our features, these will be: Unit_Name, Content_Name, and Unit_Sub_Unit_name
df['Unit_Name'] = df['Unit_Name'].apply(preprocess_text)
df['Content_Name'] = df['Content_Name'].apply(preprocess_text)
df['Unit_Sub_Unit_Name'] = df['Unit_Sub_Unit_Name'].apply(preprocess_text)

# Now lets use a naive bayesian classifier model and train it on these three features with targets being the target 

# First split the data 
train_df = df.iloc[1:]  

# Create a TfidfVectorizer to convert text into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['Unit_Name'] + ' ' + train_df['Content_Name'] + ' ' + train_df['Unit_Sub_Unit_Name'])
y_train = targets[1:]



model = MultinomialNB()
model.fit(X_train, y_train)





# ACCURACY CODE (IGNORE)
# Now go through each test entry and show me the difference between predicted and actual 
# for i, pred in enumerate(y_pred):
#     print("Predicted:", label_encoder.inverse_transform([pred]))
#     print("Actual:", label_encoder.inverse_transform([y_test[i+156]])) # Adjusting the index for y_test
#     print()
  
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# unique_targets = np.unique(y_train)

# y_test_np = y_test.to_numpy()
# y_pred_np = y_pred
# Now i want to print the targets with the worst accuracy 
# target_accuracy = {}
# for target in unique_targets:
#     target_indices = np.where(y_test_np == target)[0]
#     target_accuracy[target] = accuracy_score(y_test_np[target_indices], y_pred_np[target_indices])

# for target, accuracy in target_accuracy.items():
#     if accuracy < 1:
#         print("Target:", label_encoder.inverse_transform([target]))
#         print("Accuracy:", accuracy)
#         print()