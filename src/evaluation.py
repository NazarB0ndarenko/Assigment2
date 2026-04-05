from sklearn.metrics import accuracy_score, f1_score

def evaluate(y_true, y_pred):

    # Calculate accuracy (percentage of correct predictions)
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate F1-score (balance between precision and recall)
    f1 = f1_score(y_true, y_pred)

    # Return both evaluation metrics
    return accuracy, f1