# Automatic Ticket Classification System - Python Script (Windows Compatible)

# ===============================
# Import Libraries
# ===============================
import json
import numpy as np
import pandas as pd
import re
import string
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle
import warnings
import os

# ===============================
# Settings
# ===============================
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# ===============================
# Load JSON file
# ===============================
json_path = r"C:\Users\venkatesh\OneDrive\Documents\gopi\mini\complaints-2021-05-14_08_16.json"

with open(json_path, 'r', encoding='utf-8') as j:
    data = json.load(j)

df = pd.json_normalize(data)

# ===============================
# Preprocess Columns
# ===============================
df = df[['_source.complaint_what_happened','_source.product','_source.sub_product']]
df = df.rename(columns={
    '_source.complaint_what_happened': 'complaint_text',
    '_source.product': 'category',
    '_source.sub_product': 'sub_category'
})

# Merge category + sub_category
df['category'] = df['category'] + '+' + df['sub_category']
df.drop(['sub_category'], axis=1, inplace=True)

# Replace empty complaints with NaN and drop them
df['complaint_text'].replace('', np.nan, inplace=True)
df.dropna(subset=['complaint_text'], inplace=True)

# ===============================
# Text Cleaning Function
# ===============================
def clean_texts(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['complaint_text'] = df['complaint_text'].apply(clean_texts)

# ===============================
# Lemmatization Function
# ===============================
def lemma_texts(text):
    lemma_list = [token.lemma_ for token in nlp(text)]
    return " ".join(lemma_list)

df['lemmatized_complaint'] = df['complaint_text'].apply(lemma_texts)

# ===============================
# Extract Singular Nouns
# ===============================
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def singular_nouns(text):
    blob = TextBlob(text)
    return ' '.join([word for word, tag in blob.tags if tag == 'NN'])

df['complaint_POS_removed'] = df['lemmatized_complaint'].apply(singular_nouns)

# ===============================
# Wordcloud
# ===============================
stop_words = set(STOPWORDS)
word_cloud = WordCloud(
    background_color='white',
    stopwords=stop_words,
    max_font_size=38,
    max_words=40,
    random_state=42
).generate(' '.join(df['complaint_POS_removed']))

plt.figure(figsize=(15,12))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# ===============================
# TF-IDF and NMF Topic Modeling
# ===============================
tfidf = TfidfVectorizer(min_df=2, max_df=0.95, stop_words='english')
dtm = tfidf.fit_transform(df['complaint_POS_removed'])

num_topics = 5
nmf_model = NMF(n_components=num_topics, random_state=40)
nmf_model.fit(dtm)

# Top 20 words per topic
for index, topic in enumerate(nmf_model.components_):
    top_words = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-20:]]
    print(f'TOP 20 WORDS FOR TOPIC #{index}: {top_words}\n')

# Assign best topic to each complaint
topic_result = nmf_model.transform(dtm)
df['Topic'] = topic_result.argmax(axis=1)

# Mapping topic numbers to names
topic_mapping = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}

df['Topic'] = df['Topic'].map(topic_mapping)

# ===============================
# Prepare Training Data
# ===============================
training_data = df[['complaint_text', 'Topic']].copy()
training_data['complaint_text'] = training_data['complaint_text'].str.replace('xxxx', '')

reverse_topic_mapping = {v:k for k,v in topic_mapping.items()}
training_data['Topic'] = training_data['Topic'].map(reverse_topic_mapping)

# CountVectorizer & TF-IDF
count_vector = CountVectorizer()
X_train_count = count_vector.fit_transform(training_data['complaint_text'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

# Save vectorizers
pickle.dump(count_vector.vocabulary_, open("count_vector.pkl","wb"))
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data['Topic'], test_size=0.2, random_state=42)

# ===============================
# Train Models
# ===============================
def model_eval(y_test, y_pred, model_name):
    print(f"\nCLASSIFICATION REPORT for {model_name}:\n")
    print(classification_report(y_test, y_pred, target_names=list(topic_mapping.values())))
    
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=list(topic_mapping.values()),
                yticklabels=list(topic_mapping.values()))
    plt.title(f"CONFUSION MATRIX for {model_name}")
    plt.show()

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
f1_lr = f1_score(y_test, y_pred_lr, average="weighted")
print("F1 Score (Logistic Regression):", f1_lr)
model_eval(y_test, y_pred_lr, "LOGISTIC REGRESSION")

# Save Logistic Regression model
pickle.dump(lr, open("logreg_model.pkl", "wb"))

# ===============================
# Topic Predictor Function
# ===============================
def topic_predictor(text_list):
    vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl","rb")))
    tfidf_loaded = pickle.load(open("tfidf.pkl","rb"))
    model = pickle.load(open("logreg_model.pkl","rb"))
    
    X_new_count = vec.transform(text_list)
    X_new_tfidf = tfidf_loaded.transform(X_new_count)
    prediction = model.predict(X_new_tfidf)
    
    # Map back to topic names
    reverse_map = {v:k for k,v in reverse_topic_mapping.items()}
    return [reverse_map[p] for p in prediction]

# ===============================
# Test on Sample Complaints
# ===============================
sample_complaints = [
    "I can not get from chase who services my mortgage, who owns it and who has original loan docs",
    "The bill amount of my credit card was debited twice. Please look into the matter and resolve at the earliest.",
    "I want to open a salary account at your downtown branch. Please provide me the procedure.",
    "Unwanted service activated and money deducted automatically",
    "How can I know my CIBIL score?",
    "Where are the bank branches in the city of Patna?"
]

predicted_topics = topic_predictor(sample_complaints)
for c, t in zip(sample_complaints, predicted_topics):
    print(f"Complaint: {c}\nPredicted Topic: {t}\n")
