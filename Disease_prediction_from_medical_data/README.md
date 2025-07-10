Heart Disease Prediction Project
Overview
This project focuses on building a machine learning model to predict the presence or absence of heart disease in individuals based on various medical parameters. It serves as an introductory hands-on experience in a typical machine learning classification pipeline, from data acquisition and preprocessing to model training and evaluation.

Objective
The primary objective is to develop a classification model that can accurately predict whether an individual has heart disease (binary outcome: Yes/No) using their past medical data. This model aims to assist in early identification of high-risk individuals, potentially aiding in preventative care and timely diagnosis.

Dataset
The project utilizes the Heart Disease Data Set (specifically the processed.cleveland.data subset) from the UCI Machine Learning Repository.

Source: UCI Machine Learning Repository - Heart Disease Dataset

Key Features (Examples):

age: Age of the individual.

sex: Gender (0 = female, 1 = male).

cp: Chest pain type (e.g., typical angina, atypical angina).

trestbps: Resting blood pressure.

chol: Serum cholesterol.

fbs: Fasting blood sugar.

restecg: Resting electrocardiographic results.

thalach: Maximum heart rate achieved.

exang: Exercise induced angina.

oldpeak: ST depression induced by exercise relative to rest.

ca: Number of major vessels colored by fluoroscopy.

thal: Thalassemia (a blood disorder).

Target Variable: target (originally num), where 0 indicates no heart disease and 1 (or higher values, which are mapped to 1) indicates the presence of heart disease.

Approach
The project follows a standard machine learning workflow for a classification task:

Data Acquisition: Downloaded the processed.cleveland.data file.

Initial Data Inspection: Examined the raw data format, identified delimiters, headers, and potential missing values.

Data Preprocessing:

Handled missing values (identified as '?' and dropped corresponding rows).

Transformed the multi-class target variable (0, 1, 2, 3, 4) into a binary outcome (0 for no disease, 1 for disease).

Applied One-Hot Encoding to categorical features to ensure proper interpretation by machine learning algorithms.

Data Splitting: Divided the processed dataset into training and testing sets (80% train, 20% test) using stratify=y for balanced class distribution.

Model Training:

Trained a baseline Logistic Regression model.

Trained a Random Forest Classifier model.

Model Evaluation:

Assessed model performance using key classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Analyzed the Confusion Matrix and Classification Report to understand prediction errors (True Positives, False Negatives, etc.).

Model Improvement (Hyperparameter Tuning):

Utilized K-Fold Cross-Validation for a more robust evaluation of model performance.

Performed Grid Search to find the optimal hyperparameters for the Random Forest Classifier, specifically optimizing for Recall due to the criticality of minimizing False Negatives in medical diagnosis.

Key Features & Libraries Used
Python: The primary programming language.

Pandas: For efficient data loading, manipulation, and preprocessing (e.g., read_csv, dropna, get_dummies).

NumPy: For numerical operations.

Scikit-learn: The core machine learning library for:

train_test_split: Splitting data into training and testing sets.

LogisticRegression: Baseline classification model.

RandomForestClassifier: Ensemble classification model.

accuracy_score, precision_score, recall_score, f1_score, roc_auc_score: Model evaluation metrics.

confusion_matrix, classification_report: Detailed performance reports.

StratifiedKFold, cross_val_score: K-Fold Cross-Validation for robust evaluation.

GridSearchCV: Hyperparameter tuning.

Matplotlib & Seaborn: For data visualization (e.g., Confusion Matrix heatmap).

Results
After preprocessing and hyperparameter tuning (optimizing for Recall), the tuned Random Forest Classifier demonstrated the following performance on the unseen test set:

Accuracy: [Insert Best RF Accuracy here]

Precision: [Insert Best RF Precision here]

Recall: [Insert Best RF Recall here]

F1-Score: [Insert Best RF F1-Score here]

ROC-AUC: [Insert Best RF ROC-AUC here]

Confusion Matrix (Example Structure):

[[TN  FP]
 [FN  TP]]

Where:

TN: True Negatives (Correctly predicted no disease)

FP: False Positives (Incorrectly predicted disease)

FN: False Negatives (Incorrectly predicted no disease, but actual disease)

TP: True Positives (Correctly predicted disease)

The improved Recall after tuning indicates a better ability to correctly identify individuals who actually have heart disease, which is a critical aspect for a medical diagnostic tool.

How to Run
Download the Dataset: Obtain the processed.cleveland.data file from the UCI Machine Learning Repository - Heart Disease Dataset.

Setup Environment:

Install Python (Anaconda distribution recommended).

Install necessary libraries: pip install pandas numpy scikit-learn matplotlib seaborn jupyter (or use Google Colab).

Jupyter Notebook/Google Colab:

Open a new Jupyter Notebook or Google Colab notebook.

Upload the processed.cleveland.data file to your notebook environment.

Copy and paste the Python code cells provided during the project walkthrough into your notebook.

Run the cells sequentially to execute the data loading, preprocessing, model training, evaluation, and tuning steps.
