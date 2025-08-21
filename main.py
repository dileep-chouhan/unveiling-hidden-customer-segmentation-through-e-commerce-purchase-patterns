import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 500
data = {
    'CustomerID': range(1, num_customers + 1),
    'PurchaseFrequency': np.random.poisson(lam=5, size=num_customers), #average purchases per year
    'AverageOrderValue': np.random.normal(loc=75, scale=25, size=num_customers), #average value of each order
    'TotalSpent': np.random.normal(loc=300, scale=100, size=num_customers) #total amount spent
}
df = pd.DataFrame(data)
df = df[df['AverageOrderValue'] > 0] #remove negative values (noise)
df = df[df['TotalSpent'] > 0] #remove negative values (noise)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data, but this section is crucial for real-world datasets.
# For example, handling missing values, outliers, and data type conversions would be done here.
# --- 3. Analysis: Customer Segmentation using KMeans Clustering ---
# Select features for clustering
X = df[['PurchaseFrequency', 'AverageOrderValue']]
# Determine optimal number of clusters (e.g., using the Elbow method - uncomment to use)
# inertia = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(X)
#     inertia.append(kmeans.inertia_)
# plt.plot(range(1, 11), inertia, marker='o')
# plt.title('Elbow Method for Optimal k')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.savefig('elbow_method.png')
# print("Plot saved to elbow_method.png")
# Apply KMeans clustering with 3 clusters (based on visual inspection of the elbow method or domain knowledge)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PurchaseFrequency', y='AverageOrderValue', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segmentation based on Purchase Frequency and Average Order Value')
plt.xlabel('Purchase Frequency')
plt.ylabel('Average Order Value')
plt.savefig('customer_segmentation.png')
print("Plot saved to customer_segmentation.png")
# --- 5. Interpretation and Actionable Insights ---
# Analyze the characteristics of each cluster to understand customer segments.
# For example, calculate the average PurchaseFrequency and AverageOrderValue for each cluster.
print("\nCluster Characteristics:")
print(df.groupby('Cluster')[['PurchaseFrequency', 'AverageOrderValue', 'TotalSpent']].mean())
# Based on the identified segments, tailor marketing campaigns and strategies.
# For instance, offer loyalty programs to high-value customers (Cluster with high TotalSpent),
# and targeted promotions to increase purchase frequency for low-frequency customers.