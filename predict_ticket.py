import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json

# ----------------------------
# Load tokenizer and model
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model/tokenizer")
model = DistilBertForSequenceClassification.from_pretrained("bert_model/model")
model.eval()  # set to evaluation mode

# ----------------------------
# Load label mapping
# ----------------------------
with open("bert_model/label_map.json", "r") as f:
    label_map = json.load(f)

# reverse mapping: index -> label
index_to_label = {int(v): k for k, v in label_map.items()}

# ----------------------------
# Prediction function
# ----------------------------
def predict_ticket(text):
    enc = tokenizer([text], truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        predicted_index = torch.argmax(logits, dim=1).item()
        return index_to_label[predicted_index]

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    examples = [
        "App crashes on login",
        "Unable to reset password",
        "Payment deducted twice",
        "Website is very slow",
        "Feature request: dark mode"
    ]
    
    for text in examples:
        label = predict_ticket(text)
        print(f"Text: {text}  → Predicted Label: {label}")
