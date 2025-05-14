# MLflow Installation Guide

## Prerequisites

Make sure you have Python 3.7+ and pip installed.

Check your Python and pip versions:

```bash
python --version
pip --version
```

## Installation

### 1. Create and activate a virtual environment (optional but recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
mlflow --version
```

You should see something like:

```
mlflow, version 2.x.x
```

## Run MLflow UI (Optional)

You can run the MLflow tracking UI locally:

```bash
mlflow ui
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

## Deactivate Virtual Environment

When done, you can deactivate the environment:

```bash
deactivate
```

# Role Categorization Guide

To run the role categorization server and MLflow tracking:

### 1. Start MLflow Tracking Server

```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 2. Run Role Categorization Script

```bash
python .\mlflow\role_categorization\role_categorization_<version>.py
```

Replace `<version>` with the specific script version name.

# Task Classification Guide

Run the task classification script:

```bash
python .\mlflow\task_classification\task_classification_<version>.py
```

Replace `<version>` with the appropriate version.

# Personalized Feedback Generation Guide

Run the feedback generation script:

```bash
python .\mlflow\personalize_feedback_generation\personalize_feedback_generation_<version>.py
```

Make sure to correct the script name if there's a typo in the filename.

# API Server Guide

### 1. Navigate to the API folder:

```bash
cd api
```

### 2. Run the Flask API server:

```bash
python app.py
```

The API will typically be available at:

```
http://127.0.0.1:5000
```
