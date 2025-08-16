# Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

# 1. Data Cleaning and Preprocessing
# Load Data
loadedData = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.DataFrame(loadedData)

# Replace TotalCharges Values ' ' to '' and '' to Null
df['TotalCharges'] = df['TotalCharges'].replace(' ','')
df['TotalCharges'] = df['TotalCharges'].replace('',np.nan)

# Convert TotalCharges Data Type from Str to Float
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


# 2. Exploratory Data Analysis (EDA)
# Visualize Tenur Data
figure1 = plt.figure(figsize = (10, 6))
sns.set_style('whitegrid')
sns.histplot(df['tenure'], bins= 15, kde = True)
plt.title('Distribution of Customer Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')

# Visualize Contract Data
figure2 = plt.figure(figsize = (10,6))
sns.countplot(df['Contract'])

# Visualize Churn Data
figure3 = plt.figure(figsize = (10,6))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn Rate by Contract Type', fontsize=16)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)


# 3. Feature Engineering
# OneHotEncoder for InternetService
ohe = OneHotEncoder(handle_unknown= 'ignore', sparse_output=False).set_output(transform="pandas")
oneHotTransform = ohe.fit_transform(df[['InternetService']])
df = pd.concat([df, oneHotTransform], axis = 1)

# Mapping Label Encoding for gender and Churn
df['gender'].unique()
df['EncodedGender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})


# 4. Model Building and Training
# Train Split Test
X = df.drop(columns = ['Churn', 'gender', 'InternetService', 'customerID'], axis = 1)
y = df['Churn']
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size= 0.2, random_state= 11)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Choose and Train Model (Logisitic Regression Classifier)
logRegModel = LogisticRegression(random_state=16, max_iter=1000)
logRegModel.fit(X_train_scaled, y_train)

# Choose and Train Model (Random Forest Classifier)
randForClModel = RandomForestClassifier(random_state=42)
randForClModel.fit(X_train_scaled, y_train)

# Choose and Train Model (Gradient Boosting Classifier)
gradBoostClModel = GradientBoostingClassifier(random_state=42)
gradBoostClModel.fit(X_train_scaled, y_train)


# Accuracy Calculation
# Logisitic Regression Classifier
y_pred_logReg = logRegModel.predict(X_test_scaled)
accuracyLogReg = accuracy_score(y_pred_logReg, y_test)

# Random Forest Classifier
y_pred_randFor = randForClModel.predict(X_test_scaled)
accuracyRandFor = accuracy_score(y_pred_randFor, y_test)

# Gradient Boosting Classifier
y_pred_gradBoost = gradBoostClModel.predict(X_test_scaled)
accuracyGradBoost = accuracy_score(y_pred_gradBoost, y_test)

print(f"Accuracy of Logisitic Regression Classifier is: {accuracyLogReg}\
      \n Accuracy of Random Forest Classifier is: {accuracyRandFor}\
      \n Accuracy of Gradient Boosting Classifier is: {accuracyGradBoost} \n")


# Precision Calculation
# Logisitic Regression Classifier
precisionLogReg = precision_score(y_test, y_pred_logReg)

# Random Forest Classifier
precisionRandFor = precision_score(y_test, y_pred_randFor)

# Gradient Boosting Classifier
precisionGradBoost = precision_score(y_test, y_pred_gradBoost)

print(f"Precision of Logisitic Regression Classifier is: {precisionLogReg}\
      \n Precision of Random Forest Classifier is: {precisionRandFor}\
      \n Precision of Gradient Boosting Classifier is: {precisionGradBoost} \n")


# Recall Calculation
# Logisitic Regression Classifier
recallLogReg = recall_score(y_test, y_pred_logReg)

# Random Forest Classifier
recallRandFor = recall_score(y_test, y_pred_randFor)

# Gradient Boosting Classifier
recallGradBoost = recall_score(y_test, y_pred_gradBoost)

print(f"Recall of Logisitic Regression Classifier is: {recallLogReg}\
      \n Recall of Random Forest Classifier is: {recallRandFor}\
      \n Recall of Gradient Boosting Classifier is: {recallGradBoost} \n")


# F1-Score Calculation
# Logisitic Regression Classifier F1-Score
f1LogReg = f1_score(y_test, y_pred_logReg)

# Random Forest Classifier
f1RandFor = f1_score(y_test, y_pred_randFor)

# Gradient Boosting Classifier
f1GradBoost = f1_score(y_test, y_pred_gradBoost)

print(f"F1-Score of Logisitic Regression Classifier is: {f1LogReg}\
      \n F1-Score of Random Forest Classifier is: {f1RandFor}\
      \n F1-Score of Gradient Boosting Classifier is: {f1GradBoost} \n")



# AUC ROC Score Calculation
# Logisitic Regression Classifier
rocAucLogReg = roc_auc_score(y_test, y_pred_logReg)

# Random Forest Classifier
rocAucRandFor = roc_auc_score(y_test, y_pred_randFor)

# Gradient Boosting Classifier
rocAucGradBoost = roc_auc_score(y_test, y_pred_gradBoost)

print(f"AUC ROC Score of Logisitic Regression Classifier is: {rocAucLogReg}\
      \n AUC ROC Score of Random Forest Classifier is: {rocAucRandFor}\
      \n AUC ROC Score of Gradient Boosting Classifier is: {rocAucGradBoost} \n")


# Cofusion Matrix Calculation
# Logisitic Regression Classifier
confMatLogReg = confusion_matrix(y_test, y_pred_logReg)

# Random Forest Classifier
confMatRandFor = confusion_matrix(y_test, y_pred_randFor)

# Gradient Boosting Classifier
confMatGradBoost = confusion_matrix(y_test, y_pred_gradBoost)

print(f"Cofusion Matrix of Logisitic Regression Classifier is: {confMatLogReg}\
      \n Cofusion Matrix of Random Forest Classifier is: {confMatRandFor}\
      \n Cofusion Matrix of Gradient Boosting Classifier is: {confMatGradBoost} \n")


