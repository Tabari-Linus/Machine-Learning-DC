from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

churn_df = pd.read_csv('C:\\Users\\Shadow\\Desktop\\Learning\\AI and ML\\Datacamp\\supervise learning\\datasets\\telecom_churn_clean.csv')

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)


X_new = np.array([[30.0, 17.5],[107.0, 24.1],[213.0, 10.9]])