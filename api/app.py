from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import openai
import json
import os
from dotenv import load_dotenv
import requests
load_dotenv()
key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=key)

# Load model artifacts
kmeans = joblib.load("model/kmeans_model.pkl")
scaler = joblib.load("model/scaler.pkl")
pca = joblib.load("model/pca.pkl")

with open("artifacts/role_mapping.json", "r") as f:
    role_mapping = json.load(f)

with open("artifacts/feature_names.json", "r") as f:
    features = json.load(f)

with open("artifacts/cluster_summary.json", "r") as f:
    cluster_summary = json.load(f)
# Flask app
app = Flask(__name__)

@app.route("/")
def health_check():
    return "Team Role Clustering Model API is running"


def explain_assignment(row):
    """Explain cluster assignment for a single row with a SHAP-like textual style."""
    cluster = row["cluster"]
    role = row["assigned_role"]
    cluster_info = cluster_summary.get(str(cluster), {})

    reasoning = {
        0: "Balanced but unremarkable — contributes, but may lack initiative",
        1: "High quality and teamwork, but very slow — may hold up the group",
        2: "Works quickly but produces low quality",
        3: "Very fast but extremely low quality and teamwork — risky contributor",
        4: "Takes on everything, delivers high quality",
        5: "Reliable and collaborative, though slow-moving",
        6: "Mediocre in all areas — lacks standout traits"
    }

    if not cluster_info:
        return "Cluster information not found."

    lines = []
    for feature in ["avg_workload", "team_work", "work_speed", "overall_quality_score"]:
        diff = row[feature] - cluster_info.get(feature, 0)
        sign = "+" if diff > 0 else "–"
        lines.append(f"{sign} {feature}: {diff:+.2f} vs cluster avg")

    lines.append(f"⇒ Overall pattern matches “{role}”: {reasoning.get(cluster, 'No reasoning available')}")

    return "\n".join(lines)

@app.route("/role", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    try:
        user_name = data['user_name']
        work_load_per_day = data['work_load_per_day']
        team_work = float(data['team_work'])
        work_category = data['work_category']
        work_speed = float(data['work_speed'])
        overall_quality_score = float(data['overall_quality_score'])
        
        df = pd.DataFrame([{
            "work_load_per_day": work_load_per_day,
            "team_work": team_work,
            "work_category": work_category,
            "work_speed": work_speed,
            "overall_quality_score": overall_quality_score
        }])
        
        # preprocess 
        # find avg workload
        df["work_load_per_day"] = df["work_load_per_day"].apply(json.loads)
        df["avg_workload"] = df["work_load_per_day"].apply(lambda x: sum(x) / len(x) if x else 0)
        
        # Select and transform input features
        X = df[features]
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        clusters = kmeans.predict(X_pca)
        print(clusters[0])
        roles = role_mapping[str(clusters[0])]
        df["cluster"] = clusters[0]
        df["assigned_role"] = roles
        print(role_mapping)
        explanation = explain_assignment(df.iloc[0])
        print(df.iloc[0])
        return jsonify({
            "user_name": data['user_name'],
            "work_load_per_day" : data['work_load_per_day'],
            "team_work" : float(data['team_work']),
            "work_category" : data['work_category'],
            "work_speed" : float(data['work_speed']),
            "overall_quality_score" : float(data['overall_quality_score']),        
            "assigned_role": roles,
            "role_explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Feedback generation function
def generate_feedback(user_name, work_category, role, work_load_per_day, team_work, work_speed, overall_quality_score):
    work_load_per_day_list = eval(str(work_load_per_day))
    avg_workload = sum(work_load_per_day_list) / len(work_load_per_day_list)

    metrics = {
        "Workload": avg_workload,
        "Teamwork": team_work,
        "Speed": work_speed,
        "Quality": overall_quality_score,
    }

    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    strengths = [name for name, _ in sorted_metrics[:2]]
    improvements = [name for name, _ in sorted_metrics[-2:]]

    analyzed_data = {
        "Workload": work_load_per_day,
        "Teamwork": team_work,
        "Speed": work_speed,
        "Quality": overall_quality_score,
        "Strengths": strengths,
        "Improvements": improvements,
        "Best Task": work_category,
        "role" : role
    }

    prompt = f"""With this analyzed Data {analyzed_data} by all score is max at 100 expect speed which is average hour max at 72 hr. Act like you're talking directly to {user_name} and give an unbias personal feedback.
    Use their name and speak casually. Highlight their strengths, especially their best work category ({work_category}), 
    and mention their highest performance stats without giving exact scores. 
    Point out areas for improvement based on their weaker aspects. 
    Keep a motivational tone and end with encouragement. Give feedback in aspect of time too."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for performance reviews."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
    
# Route to get feedback by index
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data received'}), 400
    try:
        # call role api
        url = "http://127.0.0.1:5000/role"
        headers = {"Content-Type": "application/json"}
        payload = data
        response = requests.post(url, json=payload, headers=headers)
        role_data = response.json()

        # Extract values
        user_name = role_data['user_name']
        work_load_per_day = role_data['work_load_per_day']
        team_work = float(role_data['team_work'])
        work_category = role_data['work_category']
        work_speed = float(role_data['work_speed'])
        overall_quality_score = float(role_data['overall_quality_score'])
        assigned_role = role_data['assigned_role']
        role_explanation = role_data['role_explanation']
        fd = generate_feedback(user_name, work_category, assigned_role, work_load_per_day, team_work, work_speed, overall_quality_score)
        return jsonify({
            "user_name": data['user_name'],
            "work_load_per_day" : data['work_load_per_day'],
            "team_work" : float(data['team_work']),
            "work_category" : data['work_category'],
            "work_speed" : float(data['work_speed']),
            "overall_quality_score" : float(data['overall_quality_score']),        
            "assigned_role": assigned_role,
            "role_explanation": role_explanation,
            "feedback" : fd
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
