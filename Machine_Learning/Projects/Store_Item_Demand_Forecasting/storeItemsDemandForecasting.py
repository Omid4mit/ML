# %%
"""
Author        : Omid Ahmadzadeh
GitHub        : https://github.com/Omid4mit
Email         : omid4mit@gmail.com
Created       : 2025-10-07
Last Modified : 2025-10-14

Project       : Store Item Demand Forecasting Challenge (Kaggle)
Description   :
    Comprehensive pipeline for forecasting item-level sales across multiple stores using historical data.  
    Includes data preparation, time series decomposition, rolling average visualization, feature engineering,  
    model training, evaluation, and final prediction export.

Workflow Steps:
    1. Data Cleaning and Preparation
    2. Time Series EDA & Visualization
    3. Feature Engineering from Temporal Signals
    4. Model Training and Validation (Linear, RF, XGBoost, LightGBM)
    5. Full Dataset Retraining
    6. Test Prediction and CSV Export
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
from lightgbm import LGBMRegressor

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


# 4.4 Create a function to train all models

def trainAllModels(X_train, y_train):
    """Trains a dictionary of models and returns them."""

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest' : RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        'XGBoost' : XGBRegressor(random_state=42),
        'Light GBM' : LGBMRegressor(n_estimators=1000, learning_rate=0.05,num_leaves=31,n_jobs=-1,random_state=42,force_row_wise=True)
        }

    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)
    
    return models
    

# 4.5 Training train set with trainAllModels function
validatedModels = trainAllModels(X_train, y_train)


# 4.6 Use trained models to validate models with validation set
for name, model in validatedModels.items():
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Mean Absolute Error of {name} is {mae}")



# 5 Retrain Model with full training dataset
""" (I trainded model with just a portion of training dataset,
    So I want train it with full dataset) """

# 5.1 Create X and y with full training dataset
X_train_full = trainData_sp[features]
y_train_full = trainData_sp['sales']

# 5.2 Use full dataset with trainAllModels function
fullDataModels = trainAllModels(X_train_full, y_train_full)



# 6 Predict Values for Test Dataset
# 6.1 Prepare Test Data

# 6.1.1 Split test data from full data
testData_sp = fullData_fe[fullData_fe['isTest']]

# 6.1.2 Create X_test out of testData_sp
features = ['store', 'item', 'day_of_week', 'month','year',
            'sales_lag_1', 'sales_lag_7', 'sales_7d_avg']

X_test_file = testData_sp[features]
X_test_file = X_test_file.ffill().bfill()

# 6.1.3 Drop unnecessary columns
testDataPredicted = testData_sp.drop(columns= ['day_of_week', 'month','isTest',
                                               'year', 'sales_lag_1',
                                               'sales_lag_7', 'sales_7d_avg'])

# 6.1.4 Create a copy of test prediction dataframe
testDataPredicted = testDataPredicted.copy()


# 6.1.5 Create a loop to predict target values for all models
for name,model in fullDataModels.items():
    salesColumnName = f"{name} Predicted Sales"
    y_pred = model.predict(X_test_file)
    testDataPredicted[salesColumnName] = y_pred


# 6.1.6 Delete Date column from index (It could not save in CSV file as index)
testDataPredicted.reset_index(inplace=True)


# 6.1.7 Save CSV File
testDataPredicted.to_csv(output_dir / "test_predicted_t.csv", index=False)