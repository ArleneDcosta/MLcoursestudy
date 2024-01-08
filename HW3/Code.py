# %%
#Netid: aadcosta
#Seating Pin : 79
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# %%
# Load your dataset (replace 'data.csv' with your actual data file)
data = pd.read_csv('E:\MS\MS_studies\Sem_2\IntrotoML\HW3\\71-80.csv')

# %%
# Select appropriate columns
X_data = data.iloc[:, [3,4,5,6,7]]
# Replace 'target_column_name' with the actual column name you are trying to predict
y = data.iloc[:, [8]]

# %% [markdown]
# Cross Validation for selecting the appropriate distance and the required no of components for Kmeans

# %%
# Split the data into a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2,random_state=0)

# %%
# Convert y_train and y_test from pandas DataFrame to numpy array
# y_train = y_train.to_numpy().ravel()
# y_test = y_test.to_numpy().ravel()

# Define a range of K values to evaluate
k_values = range(1, 10)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0,n_init='auto' )
    kmeans.fit(X_train)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


# %% [markdown]
# The "inertia," which represents the sum of squared distances of data points to their assigned cluster centroids.

# %%
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0,n_init='auto')
kmeans.fit(X_train)

# Assign clusters to the training data
train_cluster_assignments = kmeans.predict(X_train)

# %%
train_cluster_assignments[:20]

# %%
y_train.head(20)

# %%
# Predict clusters for test data
test_cluster_assignments = kmeans.predict(X_test)

# %%
test_cluster_assignments

# %%
len(kmeans.labels_)

# %%
kmeans.cluster_centers_

# %%
y_test

# %%
predictions_neighbor = []
predictions_centroid = []
predictions_average = []
# Step 4: Calculate Predictions
for i in range(len(X_test)):
    test_point = X_test.iloc[i]
    test_cluster = test_cluster_assignments[i]
    
    # Extract data points from the same cluster in the training set
    train_data_in_cluster = X_train[train_cluster_assignments == test_cluster]
    y_data_in_cluster = y_train[train_cluster_assignments == test_cluster]
    
    # Calculate the nearest neighbor prediction
    distances = np.linalg.norm(X_train - X_test.iloc[i], axis=1)
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_prediction = y_train.iloc[nearest_neighbor_index][0]
    
    # Calculate the centroid prediction
    cluster_centroid = kmeans.cluster_centers_[test_cluster]
    
    # Find the data point in the training set that is nearest to the cluster centroid
    distances_to_centroid = np.linalg.norm(X_train - cluster_centroid, axis=1)
    nearest_to_centroid_index = np.argmin(distances_to_centroid)
    
    # Make the prediction for the 2022 citation number
    centroid_prediction = y_train.iloc[nearest_to_centroid_index][0]
    
    # Calculate the average prediction
    average_prediction = y_data_in_cluster.mean()[0]
    #print(average_prediction)
    # Append predictions to the respective lists
    predictions_neighbor.append(nearest_neighbor_prediction)
    predictions_centroid.append(centroid_prediction)
    predictions_average.append(average_prediction)

# # Step 5: Evaluate Predictions
# # Calculate the average difference magnitude for each type of prediction (1, 2, 3)
# differences_neighbor = np.abs(predictions_neighbor - test_data['cit_2022'])
# differences_centroid = np.abs(predictions_centroid - test_data['cit_2022'])
# differences_average = np.abs(predictions_average - test_data['cit_2022'])



# %%
predictions_neighbor

# %%
predictions_centroid

# %%
predictions_average


