import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PRS_calc(PRS):
    """
    Calculate PRS and determine whether the individual is at high risk.
    Uses the Clumping and Threshold (C+T) calculation method.
    The threshold for high risk is based on previous research.

    Parameters:
        PRS (float): Polygenic Risk Score.

    Returns:
        None (Prints the genetic liability based on documented studies).
    """
    if PRS < -0.268:
        print("Your Polygenic Risk Score is in the first quartile! Your genetic liability for Alzheimer's is: low")
    elif PRS > 0.1725:
        print("Your Polygenic Risk Score is in the fourth quartile! Your genetic liability for Alzheimer's is: elevated")
    else:  # Covers -0.268 <= PRS <= 0.1725
        print("Your Polygenic Risk Score is average! Your genetic liability for Alzheimer's is: average")

def parammed_PRS_classifier(PRS, threshval):
    """
    Classifies the genetic risk based on a threshold.

    Parameters:
        PRS (float): Polygenic Risk Score.
        threshval (float): Threshold value for classification.

    Returns:
        int: 1 if PRS is above the threshold (high risk), 0 otherwise.
    """
    return 1 if PRS > threshval else 0

def success_array(test_sequences, threshvals):
    """
    Computes the classification accuracy for different threshold values.

    Parameters:
        test_sequences (DataFrame): DataFrame containing patient sequences and labels.
        threshvals (list): List of threshold values to evaluate.

    Returns:
        DataFrame: DataFrame containing threshold values and corresponding classification scores.
    """
    sequence_array = np.array(test_sequences["sequence"])
    true_array = np.array(test_sequences["label"])

    scores_df = pd.DataFrame(columns=["thresval", "score"])

    for i in range(len(threshvals)):
        threshval = threshvals[i]
        prediction_array = np.array([parammed_PRS_classifier(seq, threshval) for seq in sequence_array])
        results_array = prediction_array - true_array

        success_array = 1 - (results_array)**2  # 1 for correct, 0 for incorrect
        success_score = np.mean(success_array)  # Average accuracy

        scores_df.loc[i] = [threshval, success_score]

    return scores_df

def thresh_plot(test_sequences, threshvals, output_file='output_plot.png'):
    """
    Plots classification success scores against threshold values.

    Parameters:
        test_sequences (DataFrame): DataFrame containing patient sequences and labels.
        threshvals (list): List of threshold values.
        output_file (str): Path to save the figure.
    """
    plt.figure(figsize=(12, 6))

    scores_df = success_array(test_sequences, threshvals)
    
    plt.plot(scores_df['thresval'], scores_df['score'], label="Success Score")

    plt.title('Classification Scores')
    plt.xlabel('Threshold Value')
    plt.ylabel('Classification Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()




