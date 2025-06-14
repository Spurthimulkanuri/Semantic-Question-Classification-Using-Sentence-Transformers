import joblib
from sentence_transformers import SentenceTransformer

# Load model
model = joblib.load("model.pkl")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Use same model used during training

# Basic check
def is_question(text):
    question_words = ["what", "when", "who", "is", "are", "do", "how", "where", "why", "can", "does", "did", "will"]
    return any(text.lower().startswith(q) for q in question_words)

def is_affirmation(question):
    yes_no_starts = ["is", "are", "do", "does", "did", "can", "will", "would", "could", "should", "have", "has"]
    return any(question.lower().startswith(word) for word in yes_no_starts)

# Post-processing override for ambiguous cases
def override_prediction(question, predicted_label):
    q_lower = question.lower()
    if q_lower.startswith("what time"):
        return "when"
    return predicted_label

# Predict function
def predict_question_type(question):
    if not is_question(question):
        return "unknown"
    if is_affirmation(question):
        return "affirmation"
    vector = embedder.encode([question])
    prediction = model.predict(vector)[0]
    return override_prediction(question, prediction)

# CLI interaction
if __name__ == "__main__":
    while True:
        question = input("\nEnter a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        category = predict_question_type(question)
        print(f"Predicted Type: {category}")

