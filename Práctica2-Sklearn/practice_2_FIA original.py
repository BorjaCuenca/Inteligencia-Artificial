import itertools
import statistics

import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import os

"""
EXPERIMENT PARAMETERS
"""
NUMBER_OF_FOLDS = 10
FEATURES_TO_USE = ["studytime", "failures", "schoolsup", "health", "absences"]

"""
These will determine which parameter are used in both MLP and Decision Tree models.
Switching default to True will show a model with default parameters.
"""
DEFAULT = False
TRAIN_MLP = True

"""
MULTI-LAYER PERCEPTRON PARAMETERS
"""
LAYERS = [(50,50), (100,), (100,50)]
MAX_ITER = [200, 300, 400]
FUNC = ['identity', 'logistic', 'tanh', 'relu']
LR = [0.001, 0.01, 0.1]

"""
DECISION TREE PARAMETERS
"""
CRIT = ["gini", "entropy", "log_loss"]
MAX_DEPTH = [4, 6, 8, 10, 12, None]

def print_introduction():
    print("\nWelcome! In this project, we will be working with the Student Performance dataset obtained from the UCI "
          "Machine Learning Repository.")
    print("This dataset contains information on various personal, academic, and social factors that may influence a "
          "student's academic outcomes.")
    print("For the purposes of our analysis, we will focus on a subset of 500 students to ensure fast experimentation "
          "and testing.")
    print("The goal is to predict whether a student passes or fails based on selected features.")
    print("\nThe selected features used for training the model are:")
    print("- 'studytime': Weekly study time")
    print("- 'failures': Number of past class failures")
    print("- 'schoolsup': Extra educational support (1 = yes, 0 = no)")
    print("- 'health': Current health status (1 = very bad to 5 = very good)")
    print("- 'absences': Number of school absences")
    print("\nHere is the cleaned subset of the dataset, showing only the most relevant features for our analysis:")
    print(data)
    print("\nDISCLAIMER: This code may not provide the same result every time, as some parameters, such as maximum "
          "iterations, might be insufficient and cause the model to be trained differently in each execution.")

"""
METRICS
Accuracy is not always the best metric to measure how good a trained model is, so we will get several metrics.
"""

def get_metrics(true_labels, pred_labels):
    TP = np.sum((true_labels == 1) & (pred_labels == 1))
    TN = np.sum((true_labels == 0) & (pred_labels == 0))
    FP = np.sum((true_labels == 0) & (pred_labels == 1))
    FN = np.sum((true_labels == 1) & (pred_labels == 0))

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    if TP == 0:
        recall = 0
        precision = 0
    else:
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)

    return accuracy, recall, precision, f1

"""
Cross Validation
In order to evaluate a training approach, the division of data into training and test subsets can lead to erroneous 
conclusions due to chance. 
In order to make the conclusions more reliable, there is the Cross Validation technique. In this technique, several 
training and test divisions are created, training is performed, metrics are obtained for each of them and finally 
an average of the metrics obtained is made.
"""

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

@ignore_warnings(category=ConvergenceWarning)
def cross_validation(model, data, labels, K):

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=K)

    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    for i, (train_index, test_index) in enumerate(kf.split(data)):
        # Get the train and splits data and labels.
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]

        # Fit the model.
        model.fit(train_data, train_labels)

        # Predict.
        test_prediction = model.predict(test_data)
        test_prediction = np.asarray(test_prediction)
        test_labels = np.asarray(test_labels)

        accuracy, recall, precision, f1 = get_metrics(test_labels, test_prediction)

        # Include each obtained metric into the according list.
        acc_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)

    # Obtain the average for each list of metrics.
    average_accuracy = np.mean(acc_list)
    average_recall = np.mean(recall_list)
    average_precision = np.mean(precision_list)
    average_f1 = np.mean(f1_list)

    return average_accuracy, average_recall, average_precision, average_f1

"""
LOAD DATA
Load dataset included in scikit learn. This is a binary dataset, so the label feature can only be 1 or 0.
It is loaded as a Pandas dataframe and with (data, labels) format.
"""
from ucimlrepo import fetch_ucirepo

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
data = student_performance.data.features
labels = student_performance.data.targets

"""
CLEAN DATA
Now data should be cleaned (discard or fill missing data, use some technique to balance the amount of examples 
for each label, etc).
Some data is converted to binary values in order to implement MLP correctly.
"""
data.loc[:,'schoolsup'] = data['schoolsup'].map(lambda x: 1 if x == 'yes' else 0)
labels = labels['G3'].map(lambda x: 0 if x < 10 else 1)

data = data[FEATURES_TO_USE].head(500)

print_introduction()

if TRAIN_MLP:
    """
    MULTI-LAYER PERCEPTRON IMPLEMENTATION
    """
    from sklearn.neural_network import MLPClassifier

    if DEFAULT:
        # Asigno 900 y 0 a las iter. máximas y la semilla respect. para que los resultados con los
        # parámetros por defecto sean siempre los mismos.
        model = MLPClassifier(max_iter=900, random_state=0)
        score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
        print(f"The default metrics for MPL model are: {score}")
    else:
        best_score = 0
        best_score_mean = 0
        best_param = None
        print("\nSearching for optimal models...")

        combinations = itertools.product(LAYERS, MAX_ITER, FUNC, LR)
        counter = 0

        from tqdm import tqdm
        for parameters in tqdm(combinations, desc="Testing models"):
            counter += 1
            model = MLPClassifier(hidden_layer_sizes=parameters[0], max_iter=parameters[1],
                                  activation=parameters[2], learning_rate_init=parameters[3])
            score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
            mean = statistics.mean(np.array(score))
            if mean > best_score_mean:
                print("\nNew best model found!")
                print(f"The model number {counter} has been considered optimal.\n")
                print("\nSearching for optimal models...")
                best_score = score
                best_score_mean = mean
                best_param = parameters
        print("No new optimal model found.")
        print(f"\nThe estimated optimal parameters are: {best_param}")
        print(f"Those parameters return the following metrics: {best_score}")

else:
    """
    DECISION TREE IMPLEMENTATION
    """

    from sklearn.tree import DecisionTreeClassifier

    if DEFAULT:
        model = DecisionTreeClassifier()
        score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
        print(f"The default metrics for Decision Tree model are: {score}")
    else:
        best_score = 0
        best_score_mean = 0
        best_param = None
        print("\nSearching for optimal models...")

        combinations = itertools.product(CRIT, MAX_DEPTH)
        counter = 0

        from tqdm import tqdm
        for parameters in tqdm(combinations, desc="Testing models"):
            model = DecisionTreeClassifier(criterion=parameters[0], max_depth=parameters[1])

            score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
            mean = statistics.mean(np.array(score))
            if mean > best_score_mean:
                print("\nNew best model found!")
                print(f"The model number {counter} has been considered optimal.\n")
                print("\nSearching for optimal models...")
                best_score = score
                best_score_mean = mean
                best_param = parameters
            print("No new optimal model found.")
            print(f"\nThe estimated optimal parameters are: {best_param}")
            print(f"Those parameters return the following metrics: {best_score}")

        print(f"The estimated optimal parameters are: {best_param}")
        print(f"Those parameters return the following metrics: {best_score}")