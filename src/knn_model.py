from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train):

    # Create KNN classifier with k = 5 neighbors
    # The model predicts a class based on the majority
    # label among the 5 nearest data points
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model using the training dataset
    model.fit(X_train, y_train)

    # Return the trained model
    return model