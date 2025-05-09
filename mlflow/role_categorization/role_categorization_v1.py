import os
import json
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel


EXPERIMENT_NAME = "K-means Role Categorization"
ARTIFACT_DIR = "artifacts"
TRACKING_URI = "http://127.0.0.1:8080"
CSV_INPUT = "./dataset/k-means-train-data.csv"
N_CLUSTERS = 7

# Setup MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def preprocess_data(df):
    """Preprocess the input DataFrame."""
    try:
        df["work_load_per_day"] = df["work_load_per_day"].apply(json.loads)
        df["avg_workload"] = df["work_load_per_day"].apply(lambda x: sum(x) / len(x) if x else 0)
        label_encoder = LabelEncoder()
        df["work_category_encoded"] = label_encoder.fit_transform(df["work_category"])
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        raise
    return df


def standardize_features(df, feature_cols):
    """Standardize the features using StandardScaler."""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])
    except Exception as e:
        print(f"Error in standardizing features: {e}")
        raise
    return X_scaled, scaler


def reduce_dimensionality(X_scaled):
    """Reduce dimensionality using PCA."""
    try:
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
    except Exception as e:
        print(f"Error in reducing dimensionality: {e}")
        raise
    return X_pca, pca


def find_best_k(X_pca, max_k):
    """Find the best number of clusters using silhouette score."""
    try:
        scores = {}
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_pca)
            scores[k] = silhouette_score(X_pca, labels)
    except Exception as e:
        print(f"Error in finding the best k: {e}")
        raise
    return scores


def plot_silhouette(scores):
    """Plot silhouette scores for different k values."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(list(scores.keys()), list(scores.values()), 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.grid(True)
        path = os.path.join(ARTIFACT_DIR, "silhouette_scores.png")
        plt.savefig(path)
    except Exception as e:
        print(f"Error in plotting silhouette scores: {e}")
        raise
    return path


def get_cluster_summary(df):
    """Generate a summary of the clusters."""
    try:
        profile = df.groupby("cluster")[[
            "avg_workload", "team_work", "work_speed", "overall_quality_score"
        ]].mean()
        counts = df["cluster"].value_counts().sort_index()
        summary = profile.copy()
        summary["count"] = counts
    except Exception as e:
        print(f"Error in generating cluster summary: {e}")
        raise
    return summary


class TeamRoleClusteringModel(PythonModel):
    """
    Custom model class for K-means clustering and role assignment.

    Attributes:
        kmeans (KMeans): Trained KMeans model.
        scaler (StandardScaler): StandardScaler for feature scaling.
        pca (PCA): PCA model for dimensionality reduction.
        role_mapping (dict): Mapping of cluster labels to roles.
        feature_names (list): List of feature names used for training.
    """

    def __init__(self, kmeans, scaler, pca, role_mapping, feature_names):
        self.kmeans = kmeans
        self.scaler = scaler
        self.pca = pca
        self.role_mapping = role_mapping
        self.feature_names = feature_names

    def load_context(self, context):
        """Load the model artifacts."""
        try:
            self.kmeans = joblib.load(context.artifacts["kmeans_model"])
            self.scaler = joblib.load(context.artifacts["scaler"])
            self.pca = joblib.load(context.artifacts["pca"])
            with open(context.artifacts["role_mapping"], "r") as f:
                self.role_mapping = json.load(f)
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = json.load(f)
        except Exception as e:
            print(f"Error in loading model context: {e}")
            raise

    def predict(self, context=None, model_input=None):
        """Predict the role of team members based on their features."""
        try:
            if model_input is None:
                raise ValueError("Missing model_input for prediction")
            X = model_input[self.feature_names].values
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            clusters = self.kmeans.predict(X_pca)
            return [self.role_mapping.get(c, "Unknown") for c in clusters]
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

def explain_assignment_verbose(df, cluster_summary):
    """Explain cluster assignment with a SHAP-like textual style."""
    explanations = []

    for _, row in df.iterrows():
        cluster = row["cluster"]
        role = row["assigned_role"]
        cluster_info = cluster_summary.loc[cluster]
        reasoning = {
            0: "Balanced but unremarkable — contributes, but may lack initiative",
            1: "High quality and teamwork, but very slow — may hold up the group",
            2: "Works quickly but produces low quality",
            3: "Very fast but extremely low quality and teamwork — risky contributor",
            4: "Takes on everything, delivers high quality",
            5: "Reliable and collaborative, though slow-moving",
            6: "Mediocre in all areas — lacks standout traits"
        }

        lines = []
        for feature in ["avg_workload", "team_work", "work_speed", "overall_quality_score"]:
            diff = row[feature] - cluster_info[feature]
            sign = "+" if diff > 0 else "–"
            lines.append(f"{sign} {feature}: {diff:+.2f} vs cluster avg")

        lines.append(f"⇒ Overall pattern matches “{role}”: {reasoning.get(cluster, 'No reasoning available')}")
        explanations.append("\n".join(lines))

    df["explanation"] = explanations
    return df

def main():
    """Main function to run the K-means clustering and role assignment."""
    try:
        df = pd.read_csv(CSV_INPUT)
    except Exception as e:
        print(f"Error in reading CSV input: {e}")
        return "Failed to read CSV input"

    try:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        with mlflow.start_run(run_name="K-means Role Categorization Version 1") as run:
            df = preprocess_data(df)

            features = ["avg_workload", "team_work", "work_speed", "overall_quality_score"]
            X_scaled, scaler = standardize_features(df, features)
            X_pca, pca = reduce_dimensionality(X_scaled)

            scores = find_best_k(X_pca, min(20, len(df) - 1))
            for k, score in scores.items():
                mlflow.log_metric(f"silhouette_score_k{k}", score)
            plot_path = plot_silhouette(scores)
            mlflow.log_artifact(plot_path)

            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
            df["cluster"] = kmeans.fit_predict(X_pca)
            final_score = silhouette_score(X_pca, df["cluster"])
            mlflow.log_metric("final_silhouette_score", final_score)

            role_mapping = {
                0: "Balancer",
                1: "Perfectionist",
                2: "Task finisher",
                3: "Lone Wolf",
                4: "Leader",
                5: "Helper",
                6: "Genelarist"
            }

            df["assigned_role"] = df["cluster"].map(role_mapping)
            
            # expalin result 
            cluster_summary = df.groupby("cluster")[["avg_workload", "team_work", "work_speed", "overall_quality_score"]].mean()
            df = explain_assignment_verbose(df, cluster_summary)

            model = TeamRoleClusteringModel(
                kmeans=kmeans, scaler=scaler, pca=pca,
                role_mapping=role_mapping, feature_names=features
            )

            sample_input = df[features].head()
            sample_output = model.predict(model_input=sample_input)
            signature = infer_signature(sample_input, sample_output)

            joblib.dump(kmeans, os.path.join(ARTIFACT_DIR, "kmeans_model.pkl"))
            joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
            joblib.dump(pca, os.path.join(ARTIFACT_DIR, "pca.pkl"))
            with open(os.path.join(ARTIFACT_DIR, "role_mapping.json"), "w") as f:
                json.dump(role_mapping, f)
            with open(os.path.join(ARTIFACT_DIR, "feature_names.json"), "w") as f:
                json.dump(features, f)

            mlflow.pyfunc.log_model(
                python_model=model,
                artifact_path="team_role_model",
                signature=signature,
                input_example=sample_input,
                registered_model_name="TeamRoleClusteringModel",
                artifacts={
                    "kmeans_model": os.path.join(ARTIFACT_DIR, "kmeans_model.pkl"),
                    "scaler": os.path.join(ARTIFACT_DIR, "scaler.pkl"),
                    "pca": os.path.join(ARTIFACT_DIR, "pca.pkl"),
                    "role_mapping": os.path.join(ARTIFACT_DIR, "role_mapping.json"),
                    "feature_names": os.path.join(ARTIFACT_DIR, "feature_names.json")
                }
            )

            # Log Parameters
            mlflow.log_param("n_clusters", N_CLUSTERS)
            mlflow.log_param("features", features)
            mlflow.log_param("pca_components", pca.n_components_)
            mlflow.log_param("input_file", CSV_INPUT)
            mlflow.log_param("experiment_name", EXPERIMENT_NAME)

            df[["id", 
                "user_name",
                "work_load_per_day", 
                "team_work", 
                "work_category", 
                "work_speed", 
                "overall_quality_score", 
                "assigned_role",
                "explanation"]].to_csv("output-data/kmeans_role_categorization_assignments.csv", index=False)
            
            mlflow.log_artifact("output-data/kmeans_role_categorization_assignments.csv")

    except Exception as e:
        print(f"Error during the main execution: {e}")
        raise

if __name__ == "__main__":
    main()
