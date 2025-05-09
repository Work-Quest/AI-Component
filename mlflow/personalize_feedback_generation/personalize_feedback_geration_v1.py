import pandas as pd
import mlflow
import openai
import json
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=key)
csv_path = "output-data/kmeans_role_categorization_assignments.csv"
df = pd.read_csv(csv_path)

def generate_feedback(row):
    user_name = row["user_name"]
    work_category = row["work_category"]
    role = row["assigned_role"]
    work_load_per_day_str = row["work_load_per_day"]
    # Convert string to list using eval and calculate the average
    work_load_per_day_list = eval(work_load_per_day_str)
    avg_workload = sum(work_load_per_day_list) / len(work_load_per_day_list)

    metrics = {
        "Workload": avg_workload,
        "Teamwork": row["team_work"],
        "Speed": row["work_speed"],
        "Quality": row["overall_quality_score"],
    }

    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    strengths = [name for name, _ in sorted_metrics[:2]]
    improvements = [name for name, _ in sorted_metrics[-2:]]

    analyzed_data = {
        "Workload": row["work_load_per_day"],
        "Teamwork": row["team_work"],
        "Speed": row["work_speed"],
        "Quality": row["overall_quality_score"],
        "Strengths": strengths,
        "Improvements": improvements,
        "Best Task": work_category
    }

    prompt = f"""With this analyzed Data {analyzed_data} by all score is max at 100  .Act like you're talking directly to {user_name} and give an unbias personal feedback. 
    Use their name and speak casually. Highlight their strengths, especially their best work category ({work_category}), 
    and mention their highest performance stats without giving exact scores. 
    Point out areas for improvement based on their weaker aspects. 
    Keep a motivational tone and end with encouragement. give feedback in aspect of time too """

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

# MLflow tracking
with mlflow.start_run(run_name="KMeans_PCA_GPT_Pipeline"):
    mlflow.log_param("model", "KMeans + PCA + GPT")
    
    target_index = 5  
    row = df.iloc[target_index]
    fb = generate_feedback(row)   
    print(row) 
    print(fb)
    output_csv = "output-data/feedback_generate.csv"

print("Pipeline complete. Feedback logged in MLflow.")
