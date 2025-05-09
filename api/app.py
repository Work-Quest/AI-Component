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
    feature_names = json.load(f)

# Flask app
app = Flask(__name__)

@app.route("/")
def health_check():
    return "Team Role Clustering Model API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Select and transform input features
        X = df[feature_names]
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        clusters = kmeans.predict(X_pca)
        roles = [role_mapping.get(c, "Unknown") for c in clusters]

        return jsonify({
            "cluster": int(clusters[0]),
            "assigned_role": roles[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
