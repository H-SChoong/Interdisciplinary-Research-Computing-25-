import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parammed_PRS_classifier(PRS, threshold):
    """
    Classifies PRS scores based on a threshold.

    Parameters:
        PRS (float): Polygenic Risk Score.
        threshold (float): Threshold value for classification.

    Returns:
        int: 1 if PRS is above the threshold (high risk), 0 otherwise.
    """
    return int(PRS > threshold)

def evaluate_PRS_classification(df, threshold):
    """
    Classifies PRS scores and evaluates classification accuracy.

    Parameters:
        df (DataFrame): DataFrame containing 'PRS' and 'true_label'.
        threshold (float): PRS threshold value for classification.

    Returns:
        float: Classification accuracy (success score).
    """
    df["predicted_label"] = df["PRS"].apply(lambda x: parammed_PRS_classifier(x, threshold))
    success_score = (df["predicted_label"] == df["true_label"]).mean()
    return success_score

def plot_success_vs_threshold(file_path, sheet_name):
    """
    Loops over a list of threshold values and plots success score vs. threshold.

    Parameters:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet containing data.
    """
    # Load Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Generate threshold values from -0.5 to 0.5 in 40 steps, plus additional extreme points
    threshold_range = np.linspace(-0.5, 0.5, 40).tolist() + [-1, -0.75, 0.75, 1]

    # Compute success scores for each threshold
    success_scores = [evaluate_PRS_classification(df, threshold) for threshold in threshold_range]

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(threshold_range, success_scores, marker='o', linestyle='-', color='b', label='Success Score')
    plt.xlabel('Threshold Value')
    plt.ylabel('Classification Success Score')
    plt.title('PRS Classification Success vs. Threshold')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


plot_success_vs_threshold(file_path, sheet_name)

