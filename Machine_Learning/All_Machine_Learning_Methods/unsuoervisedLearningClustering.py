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
