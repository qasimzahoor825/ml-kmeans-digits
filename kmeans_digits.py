# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load the digits dataset
digits = load_digits()
x = digits.data
y = digits.target  

# Create the KMeans model
kmeans = KMeans(n_clusters=10, random_state=42)

# Fit model to the data
clusters = kmeans.fit_predict(x)

# Print shape of cluster centers
print("Cluster centres shape:", kmeans.cluster_centers_.shape)

# Apply PCA for 2D visualization
pca = PCA(2)
x_pca = pca.fit_transform(x)

# Plot the clusters
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, cmap='tab10', s=15)

# Plot cluster centers
plt.scatter(
    pca.transform(kmeans.cluster_centers_)[:, 0],
    pca.transform(kmeans.cluster_centers_)[:, 1],
    c='red',
    marker='x',
    s=100,
    label='centre'
)

plt.title("K-Means Clustering on Digits Dataset")
plt.legend()
plt.show()
