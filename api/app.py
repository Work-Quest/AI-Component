from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

if __name__ == "__main__":
    app.run(debug=True)
