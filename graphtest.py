import matplotlib as plt

def PRS_calc(PRS):
    """
        Calculate PRS and then uses a certain threshold to determine wether teh individual is high risk.
        Used Clumping and Threshold calulcation method, Otherwise known as PRS(C+T) method.
        Used previous research to determine the threshold of 0.00009735 PRS.
        Parameters:
        PRS: Polygenic Risk Score (Ammended by chromloc function)

        Returns:
        String: Genetic liability based on Normal Distribution documented in previous studies.


    """
    if PRS < -0.268:
      print("Your Polygenic Risk Score is in the first quartile! Your genetic liability for Alzheimer's is: low")
    if PRS > 0.1725:
      print("Your Polygenic Risk Score is in the fourth quartile! Your genetic liability for Alzheimer's is: elevated")
    if 0.1725 > PRS > -0.268:
      print("Your Polygenic Risk Score is average! Your genetic liability for Alzheimer's is: average")

def parammed_PRS_classifier(PRS, threshval):
    """
        Calculate PRS and then uses a variable threshold to determine whether the individual is high risk.
        Used Clumping and Threshold calulcation method, Otherwise known as PRS(C+T) method.
        Used previous research to determine the threshold of 0.00009735 PRS.
        Parameters:
        PRS: Polygenic Risk Score (Ammended by chromloc function)

        Returns:
        String: Genetic liability based on Normal Distribution documented in previous studies.


    """
    bin_val = 0
    if PRS > threshval:
      bin_val = 1
    else:
       bin_val = bin_val
    
    return bin_val

def success_array(test_sequences, threshvals):
   """
        Given a dataframe of patient sequences which we are using as a test set, iterate through each
        sequence in the list, and for each threshval, calculate the boolean success of the classification
        function, giving 1 if the prediction aligns with the correct value on the test set, and giving 0
        if the prediction is wrong. Then, graph the final successes on the same graph by threshold value.

    """
   sequence_array = np.array(test_sequences["sequence"])
   true_array = np.array(test_sequences["label"])
   PRS_array = PRS_calc(sequence_array)
   accuracy_list = list()
   scores_df = pd.DataFrame(columns=["thresval", "score"])

   for i in range threshvals:
      scores_df.threshval[i] = threshvals[i]
      threshval = threshvals[i]

      prediction_array = parammed_PRS_classifier(PRS_array, threshval)
      results_array = prediction_array - true_array

      success_array = 1-(results_array)**2
      success_sum = 0 
      for i in range(0,len(success_array)):
         success_sum += success_array[i]
      success_score = success_sum/ len(success_array)
      
      scores_df.score[i] = success_score

    return scores_df

def thresh_plot(test_sequences, threshvals, output_file='output_plot.png'):
    """
    Plots the success score against the set threshold value

    Parameters:
        sequences: list of patient sequences in test set
        threshvals: list of threshold values for the x axis
        output_file (str): Path to save the figure.
    """
    plt.figure(figsize=(12, 6))

    scores_df = success_array(test_sequences,threshvals)
    
    plt.plot(scores_df['threshvals'], combined_df['scores'])

    plt.title('Classification Scores')
    plt.xlabel('Threshold Value')
    plt.ylabel('Classification Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()



