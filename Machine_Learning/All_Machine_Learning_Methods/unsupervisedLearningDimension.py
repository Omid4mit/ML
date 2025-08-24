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
