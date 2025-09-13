# Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# 1. Data Cleaning and Preprocessing

# Load Data
loadTrainData = pd.read_csv(Path(__file__).parent / "train.csv")
# loadTrainData = pd.read_csv("train.csv")
loadTestData = pd.read_csv(Path(__file__).parent / "test.csv")
# loadTestData = pd.read_csv("test.csv")
df_train = pd.DataFrame(loadTrainData)
df_test = pd.DataFrame(loadTestData)

# Find Null Values in Train Data
df_train['1stFlrSF'].isnull().sum()

# Find Null Values
mask = df_train.map(lambda x: x == '' or x == ' ')

# Fill Null ('' or ' ') Values with np.na
df_train.replace({'': np.nan, ' ': np.nan}, inplace= True)
df_test.replace({'': np.nan, ' ': np.nan}, inplace= True)

for col in df_train.columns:
    if df_train[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df_train[col]):
            df_train[col] = df_train[col].fillna(df_train[col].median())
        else:
            df_train[col] = df_train[col].fillna('Unknown')

for col in df_test.columns:
    if df_test[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df_test[col]):
            df_test[col] = df_test[col].fillna(df_test[col].median())
        else:
            df_test[col] = df_test[col].fillna('Unknown')



# Set Output Path for Saving Files
output_dir = Path(__file__).parent

# 2. Exploratory Data Analysis (EDA)

# Visualize Sales Price Data
# Histogram Plot
fig1 = plt.figure(figsize = (10,6))
sns.set_style('whitegrid')
sns.histplot(df_train['SalePrice'], bins= 15, kde= True)
plt.title('Sales Price per Counts')
plt.xlabel('Counts')
plt.ylabel('Sales Price')
fig1.savefig(output_dir / "Histogram.png")


# Heatmap Plot
subset_train = df_train.filter(items= ['SalePrice', 'LotArea', 'LotFrontage', 'OverallCond', 'YearBuilt'])
fig2 = plt.figure(figsize= (10,6))
sns.heatmap(data = subset_train, cmap='coolwarm')
fig2.savefig(output_dir / "Heatmap.png")


# Boxplot SalePrice by Neighborhood
fig3 = plt.figure(figsize= (10, 6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df_train)
fig3.savefig(output_dir / "Boxplot_SalesByNeighborhood.png")


# Boxplot SalePrice by OverallQual
fig4 = plt.figure(figsize= (10,6))
sns.boxplot(x = 'OverallQual', y='SalePrice', data = df_train)
fig4.savefig(output_dir / "Boxplot_SalesByOverallQual.png")


# Boxplot SalePrice by HouseStyle
fig5 = plt.figure(figsize= (10, 6))
sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = df_train)
fig5.savefig(output_dir / "Boxplot_SalesByHouseStyle.png")


# 3. Feature Engineering

# Create New Features
# Create HouseAge Column
df_train['HouseAge'] = df_train['YrSold'] - df_train['YearBuilt']
df_test['HouseAge'] = df_test['YrSold'] - df_test['YearBuilt']

# Create TotalSqFt Column
df_train['TotalSqFt'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
df_test['TotalSqFt'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']


# 4. Model Building and Training
# Train Test Split
train_ID = df_train['Id']
test_ID = df_test['Id']
y_train  = df_train['SalePrice']
df_train = df_train.drop(['Id', 'SalePrice'], axis=1)
df_test = df_test.drop('Id', axis=1)
combined_data = pd.concat([df_train, df_test], axis=0)
combined_data_encoded = pd.get_dummies(combined_data, drop_first=True)
X = combined_data_encoded.iloc[:len(df_train)]
X_real_test = combined_data_encoded.iloc[len(df_train):]


# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)


# 4. Model Building and Training

# Choose and Select Model (Logistic Regression)
logRegModel = LinearRegression()
logRegModel.fit(X_train_scaled, y_train)


# Logistic Regression Prediction
X_real_test_scaled = scaler.transform(X_real_test)
y_pred_logReg = logRegModel.predict(X_real_test_scaled)


# Calculate R2 Score for Linear Regression by using holdout values which mined from train data
X_train_holdout, X_test_holdout, y_train_holdout, y_test_holdout = train_test_split(X, y_train, test_size= 0.2, random_state= 11)
X_test_holdout_scaled = scaler.transform(X_test_holdout)
y_pred_holdout = logRegModel.predict(X_test_holdout_scaled)
r2ScoreLogReg = r2_score(y_test_holdout, y_pred_holdout)

# Lasso Regression Model
lassoReg = LassoCV(cv= 4, max_iter= 1000)
lassoReg.fit(X_train_scaled, y_train)

# Lasso Regression Prediction
y_pred_lassoReg = lassoReg.predict(X_real_test_scaled)

# Calculate R2 Score for Lasso Regression by using holdout values which mined from train data
y_pred_holdout_lasso = lassoReg.predict(X_test_holdout_scaled)
r2ScoreLassoReg = r2_score(y_pred_holdout_lasso, y_pred_holdout)


# Ridge Regression Model
ridgeReg = RidgeCV(cv= 2)
ridgeReg.fit(X_train_scaled, y_train)


# Ridge Regression Prediction
y_pred_ridgeReg = ridgeReg.predict(X_real_test_scaled)

# Calculate R2 Score for Ridge Regression by using holdout values which mined from train data
y_pred_holdout_ridge = ridgeReg.predict(X_test_holdout_scaled)
r2ScoreRidgeReg = r2_score(y_pred_holdout_ridge, y_pred_holdout)


# Save Prediction File
df_test_predicted = df_test
df_test_predicted['PredictedSalePrice_LinReg'] = y_pred_logReg
df_test_predicted['PredictedSalePrice_Lasso'] = y_pred_lassoReg
df_test_predicted['PredictedSalePrice_Ridge'] = y_pred_ridgeReg
df_test_predicted['Id'] = test_ID

# Reorder columns to place 'Id' first
cols = ['Id'] + [col for col in df_test_predicted.columns if col != 'Id']
df_test_predicted = df_test_predicted[cols]

# Save CSV File
df_test_predicted.to_csv(output_dir / "predicted.csv", index=False)

print(f"Linear Regression R2 Score: {r2ScoreLogReg}")
print(f"Lasso Regression R2 Score: {r2ScoreLassoReg}")
print(f"Ridge Regression R2 Score: {r2ScoreRidgeReg}")