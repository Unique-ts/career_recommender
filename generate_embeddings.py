import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load data
df = pd.read_csv('dataset.csv')

# Combine relevant text features
df['combined'] = df[['description', 'skills_required', 'educational_path', 'work_style', 'personality_fit']].astype(str).agg(' '.join, axis=1)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(df['combined'].tolist())

# Save embeddings and dataframe
with open('career_embeddings.pkl', 'wb') as f:
    pickle.dump((df, embeddings), f)

print("✅ Embeddings generated and saved.")
