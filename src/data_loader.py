from sklearn.datasets import load_breast_cancer

def load_data():

    # Load dataset from sklearn
    data = load_breast_cancer()

    # Return feature matrix and target labels
    return data.data, data.target