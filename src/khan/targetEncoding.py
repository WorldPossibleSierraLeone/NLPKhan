import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.textProcesser import *



# load data and get all the unique unit names
df = pd.read_csv('../../data_sets/khan_data/khan_train_data.csv')

label_encoder = LabelEncoder()
df['SL_Unit_Name'] = df['SL_Unit_Name'].apply(preprocess_text)
df['SL_Unit_Name'] = label_encoder.fit_transform(df['SL_Unit_Name'])
targets = df['SL_Unit_Name']

print("Unique targets:", np.unique(targets))
print("length of targets df is:", len(targets))