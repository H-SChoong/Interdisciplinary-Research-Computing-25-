import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parammed_PRS_classifier(PRS, threshold):
    """Classifies PRS scores based on a threshold."""
    return int(PRS > threshold)

def evaluate_PRS_classification(df, threshold):
    """Classifies PRS scores and evaluates classification accuracy."""
    df["Predicted Label"] = df["PRS"].apply(lambda x: parammed_PRS_classifier(x, threshold))
    return (df["Predicted Label"] == df["True Label"]).mean()

def evaluate_squared_PRS_classification(df, threshold):
    """Classifies squared PRS scores and evaluates classification accuracy."""
    df["Squared PRS"] = df["PRS"].astype(float) ** 2  # Ensure column is created
    df["Predicted Label Squared"] = df["Squared PRS"].apply(lambda x: parammed_PRS_classifier(x, threshold))
    return (df["Predicted Label Squared"] == df["True Label"]).mean()

def plot_combined_success_vs_threshold():
    """Plots classification success scores for both PRS and squared PRS on the same graph."""
    df = pd.read_excel(r"C:\Users\H-SCh\OneDrive\Desktop\IRC\PRS Scores.xlsx")

    # Convert PRS column to float
    df["PRS"] = df["PRS"].astype(float)

    # Ensure Squared PRS column is created
    df["Squared PRS"] = df["PRS"] ** 2  

    # Debugging: Print column names to verify
    print("Columns in DataFrame:", df.columns)

    # Generate threshold values: PRS uses [-1, 1], Squared PRS uses [0, max_squared]
    threshold_range_prs = np.linspace(-1.0, 1.0, 100).tolist()
    threshold_range_squared = np.linspace(0, 1.0, 50).tolist()  # Only non-negative

    # Compute success scores
    success_scores_prs = [evaluate_PRS_classification(df, threshold) for threshold in threshold_range_prs]
    success_scores_squared = [evaluate_squared_PRS_classification(df, threshold) for threshold in threshold_range_squared]

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(threshold_range_prs, success_scores_prs, marker='o', linestyle='-', color='b', markersize=4, label='Success Score (PRS)')
    plt.plot(threshold_range_squared, success_scores_squared, marker='o', linestyle='--', color='r', markersize=4, label='Success Score (Squared PRS)')

    plt.xlabel('Threshold Value')
    plt.ylabel('Classification Success Score')
    plt.title('CLassification Success Scores for Naive Benchmarking Algorithms')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


plot_combined_success_vs_threshold()
