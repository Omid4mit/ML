"""
Author: Omid Ahmadzadeh  
GitHub: https://github.com/Omid4mit  
Email: omid4mit@gmail.com  
Date Created: 2025-08-02  
Last Modified: 2025-09-28  

Description:
    This script performs data cleaning, feature engineering, exploratory analysis, and classification modeling
    to predict customer churn using the Telco Customer Churn dataset:

    - Dataset: "WA_Fn-UseC_-Telco-Customer-Churn.csv" from IBM Sample Data
    - Step 1: Data Cleaning and Preprocessing
        - Handle missing and inconsistent values
        - Encode categorical variables
        - Scale numerical features
    - Step 2: Exploratory Data Analysis (EDA)
        - Visualize tenure distribution, contract types, and churn rates
    - Step 3: Feature Engineering
        - Create encoded features for gender and churn
    - Step 4: Model Building and Evaluation
        - Train Logistic Regression, Random Forest, and Gradient Boosting classifiers
        - Evaluate models using accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix

"""


# Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from pathlib import Path

# 1. Data Cleaning and Preprocessing
# Load Data
# loadedData = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
loadedData = pd.read_csv(Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.DataFrame(loadedData)

# Replace TotalCharges Values ' ' to '' and '' to Null
df['TotalCharges'] = df['TotalCharges'].replace(' ','')
df['TotalCharges'] = df['TotalCharges'].replace('',np.nan)

# Convert TotalCharges Data Type from Str to Float
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


# 2. Exploratory Data Analysis (EDA)
# Visualize Tenur Data
fig1 = plt.figure(figsize = (10, 6))
sns.set_style('whitegrid')
sns.histplot(df['tenure'], bins= 15, kde = True)
plt.title('Distribution of Customer Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')

# Visualize Contract Data
fig2 = plt.figure(figsize = (10,6))
sns.countplot(df['Contract'])

# Visualize Churn Data
fig3 = plt.figure(figsize = (10,6))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn Rate by Contract Type', fontsize=16)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)


# 3. Feature Engineering
# Mapping Label Encoding for gender and Churn
df['gender'].unique()
df['EncodedGender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})


# 4. Model Building and Training
# Train Split Test
X = df.drop(columns = ['Churn', 'gender', 'InternetService', 'customerID'], axis = 1)
y = df['Churn']

# Feature Engineering (Encoding) with pd.get_dummies
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size= 0.2, random_state= 11)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose and Train Model (Logisitic Regression Classifier)
logRegModel = LogisticRegression(random_state=16, max_iter=1000)
logRegModel.fit(X_train_scaled, y_train)

# Choose and Train Model (Random Forest Classifier)
randForClModel = RandomForestClassifier(random_state=42)
randForClModel.fit(X_train_scaled, y_train)

# Choose and Train Model (Gradient Boosting Classifier)
gradBoostClModel = GradientBoostingClassifier(random_state=42)
gradBoostClModel.fit(X_train_scaled, y_train)

models = {'Logisitic Regression Classifier' : logRegModel,
          'Random Forest Classifier' : logRegModel,
          'Gradient Boosting Classifier' : gradBoostClModel}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracyScore = accuracy_score(y_test, y_pred)
    precisionScore = precision_score(y_test, y_pred)
    recallScore = recall_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)
    rocAucScore = roc_auc_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    print(f"-- {name} --")
    print(f"Accuracy: {accuracyScore}")
    print(f"Precision: {precisionScore}")
    print(f"Recall: {recallScore}")
    print(f"F1 Score: {f1Score}")
    print(f"ROC AUC : {rocAucScore}")
    print(f"Confusion Matrix: {confusionMatrix}")
    print("-" * (len(name) + 8) + "\n")

    plt.figure(figsize = (8, 6,))
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f"Confusion Matrix for {name}")
    

plt.show()