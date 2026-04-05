from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(X, y):

    # Split dataset into training (70%) and testing (30%) sets
    # random_state ensures that the split is reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Standardize features so that each feature has mean = 0 and std = 1
    # This is particularly important for distance-based algorithms like KNN
    scaler = StandardScaler()

    # Fit scaler on training data and transform it
    X_train = scaler.fit_transform(X_train)

    # Apply the same scaling transformation to test data
    X_test = scaler.transform(X_test)

    # Return prepared datasets
    return X_train, X_test, y_train, y_test