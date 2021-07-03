'''
Looking at the previous analysis we can see that data it is required to be scaled
due to the fact that far away from the Gaussian Distribution
'''
df_scaled = df.copy()
df_scaled.head()

#scaling our features
scaler = StandardScaler()
for column in df_scaled.columns[1:]:
    df_scaled[column] = scaler.fit_transform(df_scaled[column].to_frame())

# Results Of Scaling
plt.figure(figsize = (12, 8))
df_scaled.iloc[:, 1:].hist(figsize=(20, 20), color = sns.color_palette("YlGn", 1))
plt.show();

'''
RobustScaler
'''
df_robust_scaled = df.copy()
#scaling our features
scaler2 = RobustScaler()
for column in df_robust_scaled.columns[1:]:
    df_robust_scaled[column] = scaler2.fit_transform(df_robust_scaled[column].to_frame())
# Results Of Robust Scaling
plt.figure(figsize = (12, 8))
df_robust_scaled.iloc[:, 1:].hist(figsize=(20, 20), color = 'red')
plt.show();


'''
We have to take into account that there are 5 possible groups:
a) Terrible b) Poor c) Average d) Very Good e) Excellent
'''
X_robust_features = df_robust_scaled.iloc[:, 1:].values 
X_robust_features #array with values of df_robust_scaled after scaling and without user id column

#The Elbow Method
n_clusters = 15
mean_squared_distance = []
for k in range(1, n_clusters):
    kmeans = KMeans(n_clusters = k, init = "k-means++", random_state = 42)
    kmeans.fit(X_robust_features)
    mean_squared_distance.append(kmeans.inertia_)
    
    
plt.figure(figsize=(10,6))
plt.plot(range(1, n_clusters), mean_squared_distance)
plt.scatter(3, mean_squared_distance[2], s = 100, color = 'purple', label = "Potential Optimal K = 3")
plt.scatter(6, mean_squared_distance[5], s = 100, color = 'red', label = "Potential Optimal K = 6")
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia_score')
plt.legend()
plt.show();

'''
trying to find Optimal K through silhouette score
'''
n_clusters = 15
score_silhouette = []
for k in range(2, n_clusters):
    kmeans = KMeans(n_clusters = k, init = "k-means++", random_state = 42)
    kmeans.fit(X_robust_features)
    score_silhouette.append(silhouette_score(X_robust_features, kmeans.labels_))
    
plt.figure(figsize=(10,6))
plt.plot(range(2, n_clusters), score_silhouette)
plt.scatter(4, score_silhouette[2], s = 100, color = 'orange', label = "Potential Optimal K = 4")
plt.scatter(5, score_silhouette[3], s = 100, color = 'purple', label = "Potential Optimal K = 5")
plt.scatter(6, score_silhouette[4], s = 100, color = 'lightblue', label = "Potential Optimal K = 6")
plt.scatter(7, score_silhouette[5], s = 100, color = 'red', label = "Potential Optimal K = 7")
plt.xlabel('No. of Clusters')
plt.ylabel('silhouette_score')
plt.legend()
plt.show();

'''
Clustering with k = 5
'''
np.random.seed(42)
kmeans_5clusters = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

kmeans_5clusters.fit(X_robust_features)

#Predictions
predictions_5clusters = kmeans_5clusters.predict(X_robust_features)
centroids_5clusters = kmeans_5clusters.cluster_centers_
print(f"For k = 5, number of data points in each cluster:\n{pd.DataFrame(predictions_5clusters).value_counts()}")

# Predictions for 5-means assigned in Data Frame with RobustScaler()
df_robust_scaled['Cluster5'] = pd.DataFrame(predictions_5clusters)
#Predictions for 5-means assigned in Data Frame without Scaling
clusters5_df = pd.concat([df.iloc[:, 1:], pd.DataFrame({'cluster' : predictions_5clusters})], axis = 1)

#Interpretation Of 5 - Clusters using Mean Value
colnames = clusters5_df.columns[:-1]
interpretation_of_clusters = clusters5_df.groupby(['cluster'])[colnames].median().T

#Visualizations
fig, axes = plt.subplots(5, 1, figsize = (15, 19))
fig.suptitle('Mean Value for Each Category per Cluster', fontweight = 'bold', fontsize = 15)


#Cluster 0
sns.barplot(x = interpretation_of_clusters[0].values , y = interpretation_of_clusters.index, ax = axes[0])
axes[0].set_title("Cluster 0", fontweight = "bold")
#Cluster 1
sns.barplot(x = interpretation_of_clusters[1].values , y = interpretation_of_clusters.index, ax = axes[1])
axes[1].set_title("Cluster 1", fontweight = "bold")
#Cluster 2
sns.barplot(x = interpretation_of_clusters[2].values , y = interpretation_of_clusters.index, ax = axes[2])
axes[2].set_title("Cluster 2", fontweight = "bold")
#Cluster 3
sns.barplot(x = interpretation_of_clusters[3].values , y = interpretation_of_clusters.index, ax = axes[3])
axes[3].set_title("Cluster 3", fontweight = "bold")
#Cluster 4
sns.barplot(x = interpretation_of_clusters[4].values , y = interpretation_of_clusters.index, ax = axes[4])
axes[4].set_title("Cluster 4", fontweight = "bold");



'''
Clustering with k = 4
'''
np.random.seed(42)
kmeans_4clusters = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

kmeans_4clusters.fit(X_StandardScaled_feats)

#Predictions
predictions_4clusters = kmeans_4clusters.predict(X_StandardScaled_feats)
centroids_4clusters = kmeans_4clusters.cluster_centers_
print(f"For k = 4, number of data points in each cluster:\n{pd.DataFrame(predictions_4clusters).value_counts()}")

'''
Clustering with k = 6
'''
np.random.seed(42)
kmeans_6clusters = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)

kmeans_6clusters.fit(X_StandardScaled_feats)

#Predictions
predictions_6clusters = kmeans_6clusters.predict(X_StandardScaled_feats)
centroids_6clusters = kmeans_6clusters.cluster_centers_
print(f"For k = 6, number of data points in each cluster:\n{pd.DataFrame(predictions_6clusters).value_counts()}")

'''
Clustering with k = 7
'''
np.random.seed(42)
kmeans_7clusters = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)

kmeans_7clusters.fit(X_StandardScaled_feats)

#Predictions
predictions_7clusters = kmeans_7clusters.predict(X_StandardScaled_feats)
centroids_7clusters = kmeans_7clusters.cluster_centers_
print(f"For k = 7, number of data points in each cluster:\n{pd.DataFrame(predictions_7clusters).value_counts()}")

print(f"For k : 4 mean square distance between each instance and its closest centroid is {kmeans_4clusters.inertia_}")
print(f"For k : 5 mean square distance between each instance and its closest centroid is {kmeans_5clusters.inertia_}")
print(f"For k : 6 mean square distance between each instance and its closest centroid is {kmeans_6clusters.inertia_}")
print(f"For k : 7 mean square distance between each instance and its closest centroid is {kmeans_7clusters.inertia_}")
print("-" *100)
print(f"For k : 4 between  clusters'  distance is {silhouette_score(X_StandardScaled_feats, predictions_4clusters )}")
print(f"For k : 5 between  clusters'  distance is {silhouette_score(X_StandardScaled_feats, predictions_5clusters )}")
print(f"For k : 6 between  clusters'  distance is {silhouette_score(X_robust_features, predictions_6clusters )}")
print(f"For k : 7 between  clusters'  distance is {silhouette_score(X_StandardScaled_feats, predictions_7clusters )}")