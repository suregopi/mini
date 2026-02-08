import streamlit as st
import pickle

# ===============================
# Load models and vectorizers
# ===============================
count_vector = pickle.load(open("count_vector.pkl", "rb"))
tfidf_transformer = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("logreg_model.pkl", "rb"))

reverse_topic_mapping = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}

# ===============================
# Function to predict topic
# ===============================
def topic_predictor(text_list):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    vec = CountVectorizer(vocabulary=count_vector)
    X_count = vec.transform(text_list)
    X_tfidf = tfidf_transformer.transform(X_count)
    prediction = model.predict(X_tfidf)
    return [reverse_topic_mapping[p] for p in prediction]

# ===============================
# Streamlit UI
# ===============================
st.title("Automatic Ticket Classification System")
st.write("Enter a complaint text and see the predicted category/topic.")

user_input = st.text_area("Enter complaint here:")

if st.button("Predict Topic"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint text.")
    else:
        predicted_topic = topic_predictor([user_input])[0]
        st.success(f"Predicted Topic: {predicted_topic}")
