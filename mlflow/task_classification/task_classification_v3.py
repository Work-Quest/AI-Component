import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from transformers import pipeline
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Define the models to evaluate
MODELS = [
    "facebook/bart-large-mnli",  
    "cross-encoder/nli-distilroberta-base",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
]
#  v1
# Work categories to predict from
# work_categories = [
#     "Research", "Writing", "Presentation Design", "Presenting", "Planning",
#     "Programming", "Graphic Design", "Spreadsheet Work", "Problem Solving",
#     "Content Creation", "Script Writing", "Reviewing", "Documentation", "Testing",
#     "Report Formatting", "Translation", "Drawing/Illustration", "Code Review",
#     "Diagram Creation", "Flowchart Design", "Mockup Design", "Storyboarding",
#     "Email Writing", "Peer Review", "Reference Finding", "Submitting"
# ]
# v2
# Select label from confusion
# work_categories = [
#     "Research", "Presentation Design", "Planning",
#     "Programming", "Graphic Design", "Spreadsheet Work",
#     "Content Creation", "Reviewing", "Documentation", "Testing",
#     "Translation", "Drawing/Illustration", "Writing"
#     "Diagram Creation", 
#     "Email Writing", "Reference Finding", "Submitting"
# ]
# v3
# reduce number of label by group and merge form confusion and rename label
# work_categories = [
#     "Research", "Content Design", "Assign, Setup and Task",
#     "Programming", "Spreadsheet Work",
#     "Reviewing", "Documentation", "Testing",
#     "Translation",  "Email Sending", "Finalizing and Submitting Deliverables"
# ]
work_categories = [
    "Conducting Research",
    "Creating Content and Visuals",
    "Task Assignment and Scheduling",
    "Programming",
    "Working with Spreadsheets and Data",
    "Reviewing and Providing Feedback",
    "Documentation",
    "Testing",
    "Translation",
    "Sending Emails and Communication",
    "Finalizing and Submitting Work"
]

def load_data(file_path):
    """Load and prepare dataset"""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} entries")
    print(f"Columns: {df.columns}")
    
    # Ensure the CSV has necessary columns
    required_columns = ["task_description", "category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Print category distribution
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    
    return df

def evaluate_model(model_name, test_data):
    """Evaluate a specific model on test data"""
    print(f"\nEvaluating model: {model_name}")
    
    # Initialize the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model=model_name)
    
    # Predictions and ground truth
    y_true = []
    y_pred = []
    confidences = []
    
    # Process each test example
    for idx, row in test_data.iterrows():
        task = row["task_description"]
        true_category = row["category"]
        
        # Perform classification
        result = classifier(task, work_categories)
        
        # Get prediction
        predicted_category = result["labels"][0]
        confidence = result["scores"][0]
        
        y_true.append(true_category)
        y_pred.append(predicted_category)
        confidences.append(confidence)
        
        # Print progress every 10 examples
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_data)} examples")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Detailed classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create confusion matrix data
    unique_categories = sorted(list(set(y_true + y_pred)))
    confusion = pd.DataFrame(0, index=unique_categories, columns=unique_categories)
    for t, p in zip(y_true, y_pred):
        confusion.loc[t, p] += 1
    
    # Return results
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": list(zip(y_true, y_pred, confidences)),
        "confusion_matrix": confusion.to_dict(),
        "classification_report": class_report
    }
    
    return results

def plot_confusion_matrix(confusion_matrix, model_name, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt="d")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_metrics_comparison(metrics, output_path):
    """Plot and save comparison of model metrics"""
    metrics_df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=0)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load dataset
    dataset_path = "./dataset/task-category-dataset-v3.csv"
    df = load_data(dataset_path)
    df = df.dropna()

    print(f"Test set: {len(df)} examples")
    
    # Create experiment and run for the overall comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"WorkCategory_ModelComparison_{timestamp}"
    mlflow.set_experiment(experiment_name)
    
    # For storing comparative metrics
    comparison_metrics = {
        "models": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    
    # Evaluate each model
    with mlflow.start_run(run_name="model_comparison"):
        # Log dataset info
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("test_size", len(df))
        
        # Create output directory for artifacts
        output_dir = "model_comparison_results/task_classification/v3"
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each model
        for model_name in MODELS:
            # Create a child run for this model
            with mlflow.start_run(run_name=f"model_{model_name.split('/')[-1]}", nested=True) as run:
                # Log model name
                mlflow.log_param("model_name", model_name)
                
                # Evaluate model
                results = evaluate_model(model_name, df)
                
                # Log metrics
                mlflow.log_metric("accuracy", results["accuracy"])
                mlflow.log_metric("precision", results["precision"])
                mlflow.log_metric("recall", results["recall"])
                mlflow.log_metric("f1_score", results["f1_score"])
                
                # Add to comparison
                comparison_metrics["models"].append(model_name.split('/')[-1])
                comparison_metrics["accuracy"].append(results["accuracy"])
                comparison_metrics["precision"].append(results["precision"])
                comparison_metrics["recall"].append(results["recall"])
                comparison_metrics["f1_score"].append(results["f1_score"])
                
                # Save detailed results
                model_results_path = f"{output_dir}/{model_name.split('/')[-1]}_results.json"
                with open(model_results_path, "w") as f:
                    # Convert confusion matrix from dict to list for JSON serialization
                    results_json = results.copy()
                    results_json["confusion_matrix"] = dict(results["confusion_matrix"])
                    json.dump(results_json, f, indent=2)
                mlflow.log_artifact(model_results_path)
                
                # Plot confusion matrix
                confusion_path = f"{output_dir}/{model_name.split('/')[-1]}_confusion.png"
                plot_confusion_matrix(results["confusion_matrix"], model_name, confusion_path)
                mlflow.log_artifact(confusion_path)
        
        # Create comparison plots
        comparison_path = f"{output_dir}/model_comparison.png"
        plot_metrics_comparison(comparison_metrics, comparison_path)
        mlflow.log_artifact(comparison_path)
        
        # Save comparison data
        comparison_df = pd.DataFrame(comparison_metrics)
        comparison_csv_path = f"{output_dir}/model_comparison.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)
        mlflow.log_artifact(comparison_csv_path)
        
        # Determine best model
        best_model_idx = np.argmax(comparison_metrics["f1_score"])
        best_model = comparison_metrics["models"][best_model_idx]
        best_accuracy = comparison_metrics["accuracy"][best_model_idx]
        best_f1 = comparison_metrics["f1_score"][best_model_idx]
        
        # Log best model info
        mlflow.log_param("best_model", best_model)
        mlflow.log_metric("best_model_accuracy", best_accuracy)
        mlflow.log_metric("best_model_f1", best_f1)
        
        print("\n=== Model Comparison Results ===")
        print(f"Best model: {best_model}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"Best F1 score: {best_f1:.4f}")
        print("\nDetailed results:")
        print(comparison_df)

if __name__ == "__main__":
    main()