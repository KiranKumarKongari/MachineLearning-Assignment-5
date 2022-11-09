# 1. Principal Component Analysis
#     a. Apply PCA on CC dataset.
#     b. Apply k-means algorithm on the PCA result and report your observation if the silhouette score has
#        improved or not?
#     c. Perform Scaling+PCA+K-Means and report performance.

import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# cc_dataset is a dataframe where we load the csv data.
cc_dataset = pd.read_csv("C:/Users/Kiran Kumar Kongari/PycharmProjects/ML-Assignment-5/Datasets/CC.csv")
print("\nThe Dataframe is : \n", cc_dataset)

# Checking the columns having null values and displaying the resultant columns.
columnsWithNullValues = cc_dataset.isna().any()

# a. Replacing the null values with the mean
cc_dataset['CREDIT_LIMIT'] = cc_dataset['CREDIT_LIMIT'].fillna(cc_dataset['CREDIT_LIMIT'].mean())
cc_dataset['MINIMUM_PAYMENTS'] = cc_dataset['MINIMUM_PAYMENTS'].fillna(cc_dataset['MINIMUM_PAYMENTS'].mean())

# Verifying the dataframe again for null values
f = cc_dataset[cc_dataset.isna().any(axis=1)]
print('\nVerifying customer dataframe for null values again : ', f)

# dropping the CUST_ID column
customerDf = cc_dataset.drop(['CUST_ID'], axis='columns')

x = customerDf.drop('BALANCE', axis=1).values
y = customerDf['BALANCE'].values

# -------------------------------------------------------------------------------------------------------------------------------
# a. Apply PCA on CC dataset.

# Applying PCA (k=3)
pca2 = PCA(n_components=3)
principalComponents = pca2.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, customerDf[['BALANCE']]], axis=1)
print("\nThe Dataframe after applying PCA : \n", finalDf)
# -------------------------------------------------------------------------------------------------------------------------------

# b. Apply k-means algorithm on the PCA result and report your observation if the silhouette score has
#        improved or not?
# Use elbow method to find optimal number of clusters
wcss = []  # WCSS is the sum of squared distance between each point and the centroid in a cluster
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(finalDf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the above plot we can observe a sharp edge at k=2 (Number of clusters).
# Hence k=2 can be considered a good number of the cluster to cluster this data.

km = KMeans(n_clusters=2)
km.fit(finalDf)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(finalDf)

# Calculating the silhouette score for the above clustering
score = metrics.silhouette_score(finalDf, y_cluster_kmeans)
print("\nSilhouette Score for the above cluster is : ", score)

# " After applying k-means algorithm on the PCA result and observing the report/results the
#   silhouette score(i.e., 0.5329) has improved by 2% compared to the results(i.e., silhouette score of 0.5116 )
#   of k-means problem in assignment-4 "

# ---------------------------------------------------------------------------------------------------------------------------------
# c. Perform Scaling+PCA+K-Means and report performance.
print('\n----- C. Perform Scaling+PCA+K-Means and report performance.---------------\n')
# Performing Scaling
scaler = StandardScaler()
X_Scale = scaler.fit_transform(x)

# Applying PCA (k=3)
pca2 = PCA(n_components=3)
principalComponents = pca2.fit_transform(X_Scale)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, customerDf[['BALANCE']]], axis=1)
print("\nThe Dataframe after applying PCA : \n", finalDf)

# Use elbow method to find optimal number of clusters
wcss = []  # WCSS is the sum of squared distance between each point and the centroid in a cluster
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(finalDf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the above plot we can observe a sharp edge at k=2 (Number of clusters).
# Hence k=2 can be considered a good number of the cluster to cluster this data.
km = KMeans(n_clusters=2)
km.fit(finalDf)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(finalDf)

# Calculating the silhouette score for the above clustering
score = metrics.silhouette_score(finalDf, y_cluster_kmeans)
print("\nSilhouette Score for the above cluster is : ", score)


