"""
Author: Omid Ahmadzadeh  
GitHub: https://github.com/Omid4mit  
Email: omid4mit@gmail.com  
Date Created: 2025-08-02  
Last Modified: 2025-08-09  

Description:
    This script benchmarks multiple regression algorithms on a synthetic dataset generated using scikit-learn.
    It evaluates model performance using R² scores and compares predictive accuracy across regressors.

    - Dataset: Synthetic regression data (100 samples, 2 features, Gaussian noise)
    - Models Evaluated:
        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - Support Vector Regression (SVR)
        - Decision Tree Regressor
        - Random Forest Regressor
        - Gradient Boosting Regressor
        - Multi-Layer Perceptron (MLP) Regressor
    - Workflow:
        - Generate synthetic data
        - Split into training and test sets
        - Train each model and evaluate R² score

"""


# Import Libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate sample dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "SVR": SVR(kernel='rbf'),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
    "MLP Regressor": MLPRegressor()
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.2f}")
