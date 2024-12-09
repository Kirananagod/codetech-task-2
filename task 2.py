# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Load a sample customer dataset (Here we'll use a synthetic dataset for simplicity)
# You can replace it with a real-world dataset, e.g., customers' demographics, spending, etc.
data = {
    'Age': [25, 34, 45, 23, 35, 42, 33, 40, 55, 60, 65, 70, 30, 24, 58],
    'Income': [40000, 50000, 70000, 35000, 45000, 65000, 38000, 60000, 100000, 120000, 110000, 130000, 48000, 41000, 95000],
    'Spending Score': [45, 60, 50, 30, 80, 85, 75, 60, 40, 45, 30, 25, 65, 55, 70]
}

# Convert into a DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocessing
# Normalize the features (important for K-means)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 3: Applying K-means clustering
# We need to select the optimal number of clusters (k). Let's test k = 3 for simplicity.
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 4: Evaluating the clustering with silhouette score
sil_score = silhouette_score(df_scaled, df['Cluster'])
print(f'Silhouette Score: {sil_score:.3f}')

# Step 5: Visualizing the clustering result
# We'll plot the clusters on a 2D scatter plot using 'Age' and 'Income' features
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['Income'], c=df['Cluster'], cmap='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('Age')
plt.ylabel('Income')
plt.colorbar(label='Cluster')
plt.show()

# Output the DataFrame to check the cluster assignments
print("\nClustered Data:")
print(df)

# Display cluster centers (Centroids of clusters)
print("\nCluster Centers (Centroids):")
print(scaler.inverse_transform(kmeans.cluster_centers_))
