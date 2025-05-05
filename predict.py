import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# Load enriched dataset
df = pd.read_csv("dataset.csv")

def recommend_careers(user_input):
    # Predict probabilities
    probs = model.predict_proba([user_input])[0]
    top_indices = probs.argsort()[::-1][:3]  # Top 3 matches

    recommendations = []
    for idx in top_indices:
        career_row = df.iloc[idx]
        recommendations.append({
            "career": career_row["career"],
            "description": career_row["description"],
            "skills_required": career_row["skills_required"],
            "educational_path": career_row["educational_path"],
            "average_salary_india": career_row["average_salary_india"],
            "job_roles": career_row["job_roles"],
            "work_style": career_row["work_style"],
            "personality_fit": career_row["personality_fit"],
            "growth_potential": career_row["growth_potential"],
            "score": round(probs[idx] * 100, 2)
        })
    return recommendations
