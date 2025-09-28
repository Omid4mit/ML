"""
Author: Omid Ahmadzadeh  
GitHub: https://github.com/Omid4mit  
Email: omid4mit@gmail.com  
Date Created: 2025-08-02  
Last Modified: 2025-09-28  

Description:
    This script benchmarks multiple classification algorithms on a synthetic dataset generated using scikit-learn.
    It evaluates model performance using accuracy scores and compares training speed across classifiers.

    - Dataset: Synthetic binary classification data (100 samples, 5 features)
    - Models Evaluated:
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Decision Tree
        - Random Forest
        - Naïve Bayes
        - Multi-Layer Perceptron (MLP)
    - Workflow:
        - Generate synthetic data
        - Split into training and test sets
        - Train each model and evaluate accuracy
        - Measure total runtime

"""


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import time
start_time = time.time()

X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naïve Bayes": GaussianNB(),
    "MLP Classifier": MLPClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.2f}")

print("--- %s seconds ----" % (time.time() - start_time))
