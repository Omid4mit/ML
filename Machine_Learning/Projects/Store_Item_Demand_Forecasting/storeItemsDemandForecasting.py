# %%
"""
Author: Omid Ahmadzadeh
GitHub: https://github.com/Omid4mit
Email: omid4mit@gmail.com
Date Created: 2025-10-07
Last Modified: 2025-10-12

Project       : Store Item Demand Forecasting Challenge (Kaggle)  
Description   :  
    End-to-end pipeline for forecasting item-level sales across multiple stores using historical data.  
    Includes data preparation, time series analysis, feature engineering, model training, and prediction.  

Workflow Steps:  
    1. Data Cleaning and Preparation  
    2. Time Series EDA & Visualization  
    3. Feature Engineering from Temporal Signals  
    4. Model Training and Evaluation  
    5. Final Prediction and Export

"""


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# 1. Data Cleaning and Preparation

# Set Output Path for Saving Files
output_dir = Path(__file__).parent

# 1.1 Load Train Data
loadTrainCSV = pd.read_csv(Path(__file__).parent / "train.csv")
trainData = pd.DataFrame(loadTrainCSV)

# 1.2 Convert Data Types
trainData['date'] = pd.to_datetime(trainData['date'])
trainData.set_index('date', inplace = True)

# 1.3 Create isTest column for spliting data after concatnate
trainData['isTest'] = False

# 1.4 Load Test Data
loadTestCSV = pd.read_csv(Path(__file__).parent / "test.csv")
testData = pd.DataFrame(loadTestCSV).drop(columns= ['id'])

# 1.5 Convert Data Types
testData['date'] = pd.to_datetime(testData['date'])
testData.set_index('date', inplace = True)

# 1.6 Create Sales Column in Test Dataset
testData['sales'] = np.nan

# 1.7 Create isTest column for spliting data after concatnate
testData['isTest'] = True

# 1.8 Create a Concated Data Frame with Train and Test
# for Rolling Features for Fitting in Predict Model
fullData = pd.concat([trainData, testData], sort = False)
fullData = fullData.sort_values('date')


# 2. Time Series Exploratory Data Analysis (EDA) and Visualization

# 2.1 Plot Sales over Date for Store 1
trainDataStore1 = trainData[trainData['store'] == 1 ]
fig1 = plt.figure(figsize= (10, 6))
plt.plot(trainDataStore1.index, trainDataStore1['sales'])
plt.savefig(output_dir / 'store1_sales_plot.png')

# 2.2 Plot Sales over Date for Store 1
# Filter Store 1, Item 1 and year 2017
trainDataStore1_2017 = trainData[(trainData['store'] == 1) & 
                                 (trainData['item'] == 1) &
                                 (trainData.index >= '2017-01-01') & 
                                 (trainData.index <= '2017-12-31')]

# 2.3 Set date as index
ts = trainDataStore1_2017['sales']

# 2.4 Using Statsmodels to decompose time series into: trend, seasonal and residual components
decomposition = seasonal_decompose(ts, model='additive', period=12)
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.savefig(output_dir / 'Store1_trend_seasonal_residual.png')

# 2.5 Plot Sales over Date for Store 5
# Filter Store 5, Item 1 and year 2017
trainDataStore5_2017 = trainData[(trainData['store'] == 5) & 
                                 (trainData['item'] == 1) &
                                 (trainData.index >= '2017-01-01') & 
                                 (trainData.index <= '2017-12-31')]

# 2.6 Set date as index
ts = trainDataStore5_2017['sales']

# 2.7 Using Statsmodels to decompose time series into: trend, seasonal and residual components
decomposition = seasonal_decompose(ts, model='additive', period=12)
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.savefig(output_dir / 'Store5_trend_seasonal_residual.png')

# 2.8 Calculate rolling means for Store 1, Item 1
# Calculate rolling means
trainDataStore1_2017 = trainDataStore1_2017.copy()
trainDataStore1_2017['7_day_avg'] = trainDataStore1_2017['sales'].rolling(window=7).mean()
trainDataStore1_2017['30_day_avg'] = trainDataStore1_2017['sales'].rolling(window=30).mean()

# 2.9 Plot
plt.figure(figsize=(12, 6))
plt.plot(trainDataStore1_2017['sales'], label='Daily Sales', alpha=0.5)
plt.plot(trainDataStore1_2017['7_day_avg'], label='7-Day Rolling Mean', color='orange')
plt.plot(trainDataStore1_2017['30_day_avg'], label='30-Day Rolling Mean', color='green')
plt.title('Sales with Rolling Averages for Store 1, Item 1')
plt.xlabel('date')
plt.ylabel('sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / 'Store1_Item1_rolling_means.png')



# 2.10 Calculate rolling means for Store 5, Item 1
# Calculate rolling means
trainDataStore5_2017 = trainDataStore5_2017.copy()
trainDataStore5_2017['7_day_avg'] = trainDataStore5_2017['sales'].rolling(window=7).mean()
trainDataStore5_2017['30_day_avg'] = trainDataStore5_2017['sales'].rolling(window=30).mean()

# 2.11 Plot
plt.figure(figsize=(12, 6))
plt.plot(trainDataStore5_2017['sales'], label='Daily Sales', alpha=0.5)
plt.plot(trainDataStore5_2017['7_day_avg'], label='7-Day Rolling Mean', color='orange')
plt.plot(trainDataStore5_2017['30_day_avg'], label='30-Day Rolling Mean', color='green')
plt.title('Sales with Rolling Averages for Store 5, Item 1')
plt.xlabel('date')
plt.ylabel('sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / 'Store5_Item1_rolling_means.png')
plt.close()


# 3. Feature Engineering from Time

# 3.1 Create Time-Based Features

fullData['day_of_week'] = fullData.index.dayofweek       # 0 = Monday, 6 = Sunday
fullData['month'] = fullData.index.month                 # 1 to 12
fullData['year'] = fullData.index.year                   # e.g., 2025
fullData['quarter'] = fullData.index.quarter             # 1 to 4
fullData['day_of_year'] = fullData.index.dayofyear       # 1 to 365/366

# 3.2 Create Lag Features
# 3.2.1 Create a copy to avoid SettingWithCopyWarning
fullData_fe = fullData.copy()

# 3.2.2 Create Lag and Rolling Features
fullData_fe = fullData_fe.sort_values(['store', 'item', 'date'])
fullData_fe['sales_lag_1'] = fullData_fe.groupby(['store', 'item'])['sales'].shift(1)
fullData_fe['sales_lag_7'] = fullData_fe.groupby(['store', 'item'])['sales'].shift(7)
fullData_fe['sales_7d_avg'] = (
    fullData_fe
    .groupby(['store', 'item'])['sales']
    .transform(lambda x: x.rolling(window=7).mean())
)


# 3.2.3 Handle Missing Values (Backward Fill)
fullData_fe.bfill(inplace=True)

# 4. Model Building and Training

# 4.1 Split train data from full data
trainData_sp = fullData_fe[~fullData_fe['isTest']]

# 4.2 Create a time-based split. 2013-2016 to train model and the data from 2017 to validate it
train_set = trainData_sp[trainData_sp.index < '2017-01-01']
validation_set = trainData_sp[trainData_sp.index >= '2017-01-01']

# 4.3 Create features(X) and target(y)
features = ['store', 'item', 'day_of_week', 'month','year',
            'sales_lag_1', 'sales_lag_7', 'sales_7d_avg']

X_train = train_set[features]
y_train = train_set['sales']
X_val = validation_set[features]
y_val = validation_set['sales']

# 4.4 Choose and Train Model (Linear Regression)
linearRegModel = LinearRegression()
linearRegModel.fit(X_train, y_train)

# 4.5 Choose and Train Model (Gradinet Boosting Regression)
gradientBoostModel = GradientBoostingRegressor(
    n_estimators=100,     # Fewer boosting rounds
    max_depth=3,          # Shallow trees
    learning_rate=0.1,    # Standard step size
    subsample=0.8,        # Use 80% of data per tree
    random_state=42
)
gradientBoostModel.fit(X_train, y_train)

# 4.6 Choose and Train Model (Random Forest Regression)
randomForRegModel = RandomForestRegressor(
    n_estimators=100,     # Number of trees (start small for speed)
    max_depth=10,         # Limit tree depth to prevent overfitting
    max_features='sqrt',  # Use a subset of features per split
    n_jobs=-1,            # Use all CPU cores for parallel training
    random_state=42
)
randomForRegModel.fit(X_train, y_train)

# 4.7 Choose and Train Model (XGBoost)
XGBRegModel = XGBRegressor()
XGBRegModel.fit(X_train, y_train)

# 4.8 Evaluate model on the validation_set
models = {'Linear Regression': linearRegModel,
          'Gradinet Boosting Regression': gradientBoostModel,
          'Random Forest Regression': randomForRegModel,
          'XGBoost': XGBRegModel}

for name, model in models.items():
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Mean Absolute Error of {name} is {mae}")


# 5 Retrain Model with full training dataset
# (I trainded model with just a portion of training dataset,So I want train it with full dataset)

# 5.1 Create X and y with full training dataset
X_train_full = trainData_sp[features]
y_train_full = trainData_sp['sales']

# 5.2 Choose and Train Model (Linear Regression)
linearRegModel = LinearRegression()
linearRegModel.fit(X_train_full, y_train_full)

# 5.3 Choose and Train Model (Gradinet Boosting Regression)
gradientBoostModel = GradientBoostingRegressor(
    n_estimators=100,     # Fewer boosting rounds
    max_depth=3,          # Shallow trees
    learning_rate=0.1,    # Standard step size
    subsample=0.8,        # Use 80% of data per tree
    random_state=42
)
gradientBoostModel.fit(X_train_full, y_train_full)

# 5.4 Choose and Train Model (Random Forest Regression)
randomForRegModel = RandomForestRegressor(
    n_estimators=100,     # Number of trees (start small for speed)
    max_depth=10,         # Limit tree depth to prevent overfitting
    max_features='sqrt',  # Use a subset of features per split
    n_jobs=-1,            # Use all CPU cores for parallel training
    random_state=42
)
randomForRegModel.fit(X_train_full, y_train_full)

# 5.5 Choose and Train Model (XGBoost)
XGBRegModel = XGBRegressor()
XGBRegModel.fit(X_train_full, y_train_full)


# 6 Predict Values for Test Dataset

# 6.1 Prepare Test Data

# 6.1.1 Split test data from full data
testData_sp = fullData_fe[fullData_fe['isTest']]
testData_sp.head()

# 6.1.2 Create X_test out of testData_sp
features = ['store', 'item', 'day_of_week', 'month','year',
            'sales_lag_1', 'sales_lag_7', 'sales_7d_avg']

X_test_file = testData_sp[features]
X_test_file = X_test_file.ffill().bfill()
X_test_file.head()

# 6.1.3 Predict y value for test dataset (sales value) with Linrear Regression
y_pred = linearRegModel.predict(X_test_file)
y_pred

# 6.1.4 Create a loop for predicting sales value with all models
models = {'Linear Regression': linearRegModel,
          'Gradinet Boosting Regression': gradientBoostModel,
          'Random Forest Regression': randomForRegModel,
          'XGBoost': XGBRegModel}

testDataPredicted = testData_sp.drop(columns= ['day_of_week', 'month','isTest',
                                               'year', 'sales_lag_1',
                                               'sales_lag_7', 'sales_7d_avg'])
testDataPredicted = testDataPredicted.copy()

for name,model in models.items():
    salesColumnName = f"{name} Predicted Sales"
    y_pred = model.predict(X_test_file)
    testDataPredicted[salesColumnName] = y_pred


# 6.1.7 Delete Date column from index (It could not save in CSV file as index)
testDataPredicted.reset_index(inplace=True)


# 6.1.6 Save CSV File
testDataPredicted.to_csv(output_dir / "test_predicted_p.csv", index=False)