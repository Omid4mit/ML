"""
Author: Omid Ahmadzadeh  
GitHub: https://github.com/Omid4mit  
Email: omid4mit@gmail.com  
Date Created: 2025-06-11
Last Modified: 2025-06-17  

Description:
    This script benchmarks multiple clustering algorithms on a synthetic dataset generated using scikit-learn.
    It compares clustering behavior and label assignments across different unsupervised models.

    - Dataset: Synthetic blob data with 3 centers (200 samples)
    - Models Evaluated:
        - K-Means Clustering
        - Hierarchical Clustering (Agglomerative)
        - DBSCAN (Density-Based Spatial Clustering)
        - Gaussian Mixture Model (GMM)
    - Workflow:
        - Generate synthetic data
        - Apply each clustering algorithm
        - Display the first 10 predicted cluster labels for comparison

"""


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

models = {
    "K-Means": KMeans(n_clusters=3),
    "Hierarchical Clustering": AgglomerativeClustering(n_clusters=3),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
    "GMM": GaussianMixture(n_components=3)
}

for name, model in models.items():
    labels = model.fit_predict(X)
    print(f"{name}: {labels[:10]}")  # Display first 10 labels
