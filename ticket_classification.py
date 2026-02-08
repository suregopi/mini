# ==========================================
# Automatic Ticket Classification System
# HIGH ACCURACY VERSION (ML + Rules)
# ==========================================

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# TRAINING DATA (EXPANDED & BALANCED)
# ==========================================
training_data = pd.DataFrame({
    "complaint_text": [

        # -------- Bank Account --------
        "I cannot access my bank account online",
        "Unable to login to my bank account",
        "I want to update my account details",
        "How do I open a new bank account",
        "My bank account is not working",

        # -------- Credit Card --------
        "My credit card was charged twice",
        "Incorrect charges on my credit card bill",
        "Payment not reflected on my credit card statement",
        "My credit card bill is wrong",
        "Credit card transaction failed",

        # -------- Theft / Dispute --------
        "Unauthorized withdrawal from my account",
        "Someone made a transaction without my permission",
        "Fraudulent activity detected in my account",
        "I lost my credit card and need to block it",
        "Money withdrawn without authorization",

        # -------- Mortgage / Loan --------
        "My loan application is pending",
        "I need help with my mortgage documents",
        "Mortgage payment processing is delayed",
        "Loan EMI not updated",
        "Problem with home loan account"
    ],
    "topic": [
        0, 0, 0, 0, 0,     # Bank
        1, 1, 1, 1, 1,     # Credit Card
        2, 2, 2, 2, 2,     # Theft
        3, 3, 3, 3, 3      # Mortgage
    ]
})

# ==========================================
# TOPIC MAPPING
# ==========================================
topic_mapping = {
    0: "Bank Account services",
    1: "Credit card or prepaid card",
    2: "Theft/Dispute Reporting",
    3: "Mortgage/Loan"
}

# ==========================================
# VECTORIZATION (BETTER NLP)
# ==========================================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(training_data["complaint_text"])
y = training_data["topic"]

# ==========================================
# TRAIN MODEL
# ==========================================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X, y)

# SAVE MODEL
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

# ==========================================
# RULE-BASED SAFETY NET
# ==========================================
def rule_based_override(text):
    text = text.lower()

    if any(word in text for word in [
        "unauthorized", "fraud", "stolen", "lost",
        "withdrawal", "without my permission", "scam"
    ]):
        return "Theft/Dispute Reporting"

    if any(word in text for word in [
        "credit card", "bill", "charged", "statement", "payment"
    ]):
        return "Credit card or prepaid card"

    if any(word in text for word in [
        "loan", "mortgage", "emi", "home loan"
    ]):
        return "Mortgage/Loan"

    if "account" in text:
        return "Bank Account services"

    return None

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def topic_predictor(complaints):
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))

    results = []

    for text in complaints:
        rule_result = rule_based_override(text)
        if rule_result:
            results.append(rule_result)
        else:
            X_new = vectorizer.transform([text])
            pred = model.predict(X_new)[0]
            results.append(topic_mapping[pred])

    return results

# ==========================================
# LOCAL TEST
# ==========================================
if __name__ == "__main__":
    test_cases = [
        "Unauthorized withdrawal from my account",
        "My credit card bill is wrong",
        "I cannot access my bank account online",
        "My loan application is pending"
    ]

    for c, p in zip(test_cases, topic_predictor(test_cases)):
        print(f"Complaint: {c}\nPredicted Topic: {p}\n")
