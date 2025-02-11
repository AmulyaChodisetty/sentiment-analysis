import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample training data (replace with your real dataset)
train_texts = [
    "The hotel was great and the staff was friendly.",
    "I had a terrible experience. The food was bad.",
    "Amazing place! Would definitely visit again.",
    "The service was very slow and the room was dirty."
]

# Initialize and train the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(train_texts)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("✅ Vectorizer saved as 'vectorizer.pkl'")
