import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay


def main():

    # Start execution timer
    start_time = time.time()

    # 1. Load Dataset
    print("Loading dataset...")

    data = load_breast_cancer()
    X = data.data
    y = data.target

    print("Dataset shape:", X.shape)

    # 2. Split Dataset (Training / Testing)
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # 3. Feature Scaling (important for KNN)
    print("Scaling features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train KNN Model
    print("\nTraining KNN model...")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred_knn = knn.predict(X_test)

    # 5. Train Decision Tree Model
    print("Training Decision Tree model...")

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred_dt = dt.predict(X_test)

    # 6. Evaluate Models
    print("\nEvaluating models...")

    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn)

    dt_acc = accuracy_score(y_test, y_pred_dt)
    dt_f1 = f1_score(y_test, y_pred_dt)

    # 7. Create Comparison Table
    results = pd.DataFrame({
        "Algorithm": ["KNN", "Decision Tree"],
        "Accuracy": [knn_acc, dt_acc],
        "F1 Score": [knn_f1, dt_f1]
    })

    print("\nModel Comparison:")
    print(results)

    # Identify best model based on F1-score
    best_model = results.loc[results["F1 Score"].idxmax()]

    print("\nBest Model Based on F1 Score:")
    print(best_model["Algorithm"])

    # 8. Confusion Matrix Visualization
    print("\nDisplaying confusion matrices...")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
    plt.title("KNN Confusion Matrix")
    plt.show()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt)
    plt.title("Decision Tree Confusion Matrix")
    plt.show()

    # 9. Bar Chart Comparison
    print("Displaying performance comparison chart...")

    algorithms = ["KNN", "Decision Tree"]

    accuracy_values = [knn_acc, dt_acc]
    f1_values = [knn_f1, dt_f1]

    x = np.arange(len(algorithms))
    width = 0.35

    plt.figure()

    plt.bar(x - width/2, accuracy_values, width, label="Accuracy")
    plt.bar(x + width/2, f1_values, width, label="F1 Score")

    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.title("Algorithm Performance Comparison")

    plt.xticks(x, algorithms)
    plt.legend()

    plt.show()

    # 10. Execution Time
    end_time = time.time()

    print("\nExecution Time:", round(end_time - start_time, 3), "seconds")


if __name__ == "__main__":
    main()