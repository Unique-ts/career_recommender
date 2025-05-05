from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv('dataset.csv')

# Combine features into a single searchable string for each career
df['combined_text'] = df.apply(lambda row: f"{row['career']} {row['description']} {row['skills_required']} {row['educational_path']} {row['job_roles']} {row['work_style']} {row['personality_fit']}", axis=1)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for all careers
career_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.form['user_input']
        input_embedding = model.encode(user_input, convert_to_tensor=True)

        # Compute similarity scores
        similarities = util.pytorch_cos_sim(input_embedding, career_embeddings)[0]

        # Get top 2 matches
        top_matches = torch.topk(similarities, k=3)

        results = []
        for idx in top_matches.indices:
            career_data = df.iloc[int(idx)]
            results.append({
                'career': career_data['career'],
                'description': career_data['description'],
                'skills': career_data['skills_required'],
                'education': career_data['educational_path'],
                'salary': career_data['average_salary_india'],
                'roles': career_data['job_roles'],
                'work_style': career_data['work_style'],
                'personality': career_data['personality_fit'],
                'growth': career_data['growth_potential']
            })

        return render_template('result.html', results=results, query=user_input)

    except Exception as e:
        return f"Error: {e}"


    
if __name__ == '__main__':
    app.run(debug=True)
