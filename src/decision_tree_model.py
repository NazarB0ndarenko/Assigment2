from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):

    # Create Decision Tree classifier with fixed random state
    # to ensure reproducible results
    model = DecisionTreeClassifier(random_state=42)

    # Train the model using the training dataset
    model.fit(X_train, y_train)

    # Return the trained model
    return model