import os
import json
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    """Preprocess the input DataFrame using OneHotEncoder."""
    try:
        df["work_load_per_day"] = df["work_load_per_day"].apply(json.loads)
        df["avg_workload"] = df["work_load_per_day"].apply(lambda x: sum(x) / len(x) if x else 0)

        # Reshape work_category for OneHotEncoder
        work_categories = df[["work_category"]].values
        
        # Initialize and fit OneHotEncoder with sparse_output=False
        encoder = OneHotEncoder(sparse_output=False)
        onehot = encoder.fit_transform(work_categories)
        
        # Create feature names and new DataFrame with encoded values
        feature_names = encoder.get_feature_names_out(["work_category"])
        onehot_df = pd.DataFrame(onehot, columns=feature_names)
        
        df = pd.concat([df.drop("work_category", axis=1).reset_index(drop=True), 
                       onehot_df.reset_index(drop=True)], axis=1)
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
    """Plot silhouette scores for different values of k."""
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
    """Generate a summary of clusters."""
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
        """Load the model context from artifacts."""
        try:
            self.kmeans = joblib.load(context.artifacts["kmeans_model_v2"])
            self.scaler = joblib.load(context.artifacts["scaler_v2"])
            self.pca = joblib.load(context.artifacts["pca_v2"])
            with open(context.artifacts["role_mapping_v2"], "r") as f:
                self.role_mapping = json.load(f)
            with open(context.artifacts["feature_names_v2"], "r") as f:
                self.feature_names = json.load(f)
        except Exception as e:
            print(f"Error in loading model context: {e}")
            raise

    def predict(self, context=None, model_input=None):
        """Make predictions using the trained model."""
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


def main():
    """Main function to execute the K-means role categorization."""
    try:
        df = pd.read_csv(CSV_INPUT)
    except Exception as e:
        print(f"Error in reading CSV input: {e}")
        return "Failed to read CSV input"

    try:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        with mlflow.start_run(run_name="K-means Role Categorization Version 2") as run:
            df = preprocess_data(df)

            features = ["avg_workload", "team_work", "work_speed", "overall_quality_score"] + \
                       [col for col in df.columns if col.startswith("work_category_")]

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

            model = TeamRoleClusteringModel(
            kmeans=kmeans, scaler=scaler, pca=pca,
                role_mapping=role_mapping, feature_names=features
            )

            sample_input = df[features].head()
            sample_output = model.predict(model_input=sample_input)
            signature = infer_signature(sample_input, sample_output)

            joblib.dump(kmeans, os.path.join(ARTIFACT_DIR, "kmeans_model_v2.pkl"))
            joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler_v2.pkl"))
            joblib.dump(pca, os.path.join(ARTIFACT_DIR, "pca_v2.pkl"))
            with open(os.path.join(ARTIFACT_DIR, "role_mapping_v2.json"), "w") as f:
                json.dump(role_mapping, f)
            with open(os.path.join(ARTIFACT_DIR, "feature_names_v2.json"), "w") as f:
                json.dump(features, f)

            mlflow.pyfunc.log_model(
                python_model=model,
                artifact_path="team_role_model_v2",
                signature=signature,
                input_example=sample_input,
                registered_model_name="TeamRoleClusteringModel_v2",
                artifacts={
                    "kmeans_model_v2": os.path.join(ARTIFACT_DIR, "kmeans_model_v2.pkl"),
                    "scaler_v2": os.path.join(ARTIFACT_DIR, "scaler_v2.pkl"),
                    "pca_v2": os.path.join(ARTIFACT_DIR, "pca_v2.pkl"),
                    "role_mapping_v2": os.path.join(ARTIFACT_DIR, "role_mapping_v2.json"),
                    "feature_names_v2": os.path.join(ARTIFACT_DIR, "feature_names_v2.json")
                }
            )

            mlflow.log_param("n_clusters", N_CLUSTERS)
            mlflow.log_param("features", features)
            mlflow.log_param("pca_components", pca.n_components_)
            mlflow.log_param("input_file", CSV_INPUT)
            mlflow.log_param("experiment_name", EXPERIMENT_NAME)

            output_columns = ["id", "user_name", "work_load_per_day", "team_work", 
                             "work_speed", "overall_quality_score", "assigned_role"]
            
            df[output_columns].to_csv(
                "output-data/kmeans_role_categorization_assignments_v2.csv", 
                index=False
            )
            
            mlflow.log_artifact("output-data/kmeans_role_categorization_assignments_v2.csv")

    except Exception as e:
        print(f"Error during the main execution: {e}")
        raise

if __name__ == "__main__":
    main()
