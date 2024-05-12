import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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


# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_context = _create_unverified_https_context



#Training data load 
training_khan_data = pd.read_csv('data_sets/khan_train_data.csv')
objectives_data = pd.read_csv('data_sets/Math_5th_grade_objectives.csv')

# Add a none class 
none_class = pd.DataFrame([['None', '']], columns=['Unit_Name', 'Outcomes'])
objectives_data = pd.concat([objectives_data, none_class])


# Reusable functions: 
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)



stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
training_khan_data['processed_content_name'] = training_khan_data['Content_Name'].apply(preprocess_text)
training_khan_data['processed_unit_sub_unit_name'] = training_khan_data['Unit_Sub_Unit_name'].apply(preprocess_text)
training_khan_data['processed_unit_description'] = training_khan_data['Unit_Description'].apply(preprocess_text)
training_khan_data['processed_unit_name'] = training_khan_data['Unit_name'].apply(preprocess_text)
objectives_data['processed_objectives'] = objectives_data['Outcomes'].apply(preprocess_text)

# vectorize the data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_khan_data['processed_content_name'] + " " + training_khan_data['processed_unit_sub_unit_name'] + " " + training_khan_data['processed_unit_description'])
X_objectives = vectorizer.transform(objectives_data['processed_objectives'])

# Encode the target variable
le = LabelEncoder()
all_labels = pd.concat([training_khan_data['SL_Unit'], objectives_data['Unit_Name']])
le.fit(all_labels)


y_train = le.transform(training_khan_data['SL_Unit'])
y_objectives = le.transform(objectives_data['Unit_Name'])

# Fit the model with the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the objectives data
y_pred = model.predict(X_objectives)

accuracy = accuracy_score(y_objectives, y_pred)
print(f'Accuracy: {accuracy * 100}%')