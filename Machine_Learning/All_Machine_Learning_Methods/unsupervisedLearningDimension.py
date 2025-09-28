"""
Author: Omid Ahmadzadeh  
GitHub: https://github.com/Omid4mit  
Email: omid4mit@gmail.com  
Date Created: 2025-06-11  
Last Modified: 2025-06-19  

Description:
    This script compares multiple dimensionality reduction techniques on synthetic high-dimensional data.
    It demonstrates how different algorithms project data into a lower-dimensional space for visualization or modeling.

    - Dataset: Randomly generated high-dimensional data (100 samples, 20 features)
    - Models Evaluated:
        - Principal Component Analysis (PCA)
        - Truncated Singular Value Decomposition (SVD)
        - t-Distributed Stochastic Neighbor Embedding (t-SNE)
    - Workflow:
        - Generate synthetic data
        - Standardize features
        - Apply each dimensionality reduction technique
        - Display the first 5 reduced samples for each method

"""


from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.random.rand(100, 20)  # Sample high-dimensional data
X_scaled = StandardScaler().fit_transform(X)

models = {
    "PCA": PCA(n_components=2),
    "SVD": TruncatedSVD(n_components=2),
    "t-SNE": TSNE(n_components=2)
}

for name, model in models.items():
    reduced_X = model.fit_transform(X_scaled)
    print(f"{name}: {reduced_X[:5]}")  # Show first 5 reduced samples
