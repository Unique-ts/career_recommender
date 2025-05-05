import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load dataset
data = pd.read_csv('dataset.csv')

# Initialize the pre-trained transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Convert career descriptions into embeddings
career_descriptions = data['description'].tolist()
career_embeddings = model.encode(career_descriptions)

# Save the model for later use
joblib.dump(model, 'transformer_model.pkl')
joblib.dump(career_embeddings, 'career_embeddings.pkl')

print("✅ Model trained and embeddings saved!")
