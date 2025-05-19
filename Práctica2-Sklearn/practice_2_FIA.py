import random
import statistics

import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import os

"""
EXPERIMENT PARAMETERS
"""
NUMBER_OF_FOLDS = 10
FOLDER_TO_SAVE_IMAGES= "./output_images"
FEATURES_TO_USE = ["studytime", "failures", "schoolsup", "health", "absences"]

"""
These will determine which parameter are used in both MLP and Decision Tree models.
Switching default to True will show a model with default parameters.
"""
DEFAULT = True
TRAIN_MLP = True

if not os.path.exists(FOLDER_TO_SAVE_IMAGES):
    os.makedirs(FOLDER_TO_SAVE_IMAGES)

def choose_mlp_parameters():
    layers = [(50,), (50,50), (100,50), (100,100)]
    max_iter = [150, 200, 250, 300, 350]
    func = ['identity', 'logistic', 'tanh', 'relu']
    lr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    return random.choice(layers), random.choice(max_iter), random.choice(func), random.choice(lr)

def choose_tree_parameters():
    crit = ["gini", "entropy", "log_loss"]
    max_depth = [4, 6, 8, 10, 12, None]

    return random.choice(crit), random.choice(max_depth)

"""
Accuracy is not always the best metric to measure how good a trained model is, so we will get several metrics.
Following the information from https://en.wikipedia.org/wiki/Precision_and_recall, obtain Recall, Precision and F1.
"""

def get_metrics(true_labels, pred_labels):
    TP = np.sum((true_labels == 1) & (pred_labels == 1))
    TN = np.sum((true_labels == 0) & (pred_labels == 0))
    FP = np.sum((true_labels == 0) & (pred_labels == 1))
    FN = np.sum((true_labels == 1) & (pred_labels == 0))

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)

    return accuracy, recall, precision, f1

"""
Cross Validation
In order to evaluate a training approach, the division of data into training and test subsets can lead to erroneous conclusions due to chance. 
In order to make the conclusions more reliable, there is the Cross Validation technique. In this technique, several training and test divisions are created, 
training is performed, metrics are obtained for each of them and finally an average of the metrics obtained is made.
"""

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

#@ignore_warnings(category=ConvergenceWarning)
def cross_validation(model, data, labels, K):
    """
    Use KFold to implement your own cross validation for a given model instance.
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=K)

    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    for i, (train_index, test_index) in enumerate(kf.split(data)):
        #print(f"Fold {i}")
        # Get the train and splits data and labels.
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]

        # Fit the model.
        model.fit(train_data, train_labels) # Here there is a point followed by an ellipsis (...).

        # Predict.
        test_prediction = model.predict(test_data)
        # Remember to turn test_prediction and test_labels into numpy arrays if they are not.
        test_prediction = np.asarray(test_prediction)
        test_labels = np.asarray(test_labels)

        accuracy, recall, precision, f1 = get_metrics(test_labels, test_prediction)

        # Include each obtained metric into the according list.
        acc_list.append(accuracy) # Here there is a point followed by an ellipsis (...).
        recall_list.append(recall) # Here there is a point followed by an ellipsis (...).
        precision_list.append(precision) # Here there is a point followed by an ellipsis (...).
        f1_list.append(f1) # Here there is a point followed by an ellipsis (...).

        #print(f"Accuracy={accuracy}, recall={recall}, precision={precision} and F1={f1}.")

    # Obtain the average for each list of metrics.
    average_accuracy = np.mean(acc_list)
    average_recall = np.mean(recall_list)
    average_precision = np.mean(precision_list)
    average_f1 = np.mean(f1_list)

    #print(f"Provided model with cross validation K={K} gets accuracy={average_accuracy}, recall={average_recall}, precision={average_precision} and F1={average_f1}.")

    return average_accuracy, average_recall, average_precision, average_f1

"""
LOAD DATA

Load breast cancer dataset included in scikit learn. This is a binary dataset, so the label feature (cancer Yes o cancer No) can only be 1 or 0.
Follow the following example to load the data from sklearn.
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
Remember to load it as a Pandas dataframe and with (data, labels) format.
"""
from ucimlrepo import fetch_ucirepo

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
data = student_performance.data.features
data.loc[:,'schoolsup'] = data['schoolsup'].map(lambda x: 1 if x == 'yes' else 0)
labels = student_performance.data.targets
labels = labels['G3'].map(lambda x: 0 if x < 10 else 1)

#print("")
#print("List of features included in the dataset:")
#print(list(data.columns))
#print("")
#print("The table with the data looks as follows:")
#print(data)
#print(f"We have a total of {data.shape[0]} examples and {labels.shape[0]} labels with values {labels.unique()}.\n")

"""
CLEAN DATA

Now data should be cleaned (discard or fill missing data, use some technique to balance the amount of examples for each label, etc).
In this exercies, remember to filter the dataframe to use only features in FEATURES_TO_USE.
"""

data = data[FEATURES_TO_USE]
#print("The table with the CLEAN data looks as follows:")
#print(data)

if TRAIN_MLP:
    """
    Now you will instantiate the models.
    In this practice we'll train a Multilayer Perceptron Classifier and Decision Tree
    you can follow the example from the following link to instantiate a MLP:
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    Only arguments you need to worry about when initializing the model is hidden_layer_sizes and random_state. Both of them are in Section 0.
    """
    from sklearn.neural_network import MLPClassifier

    if DEFAULT:
        model = MLPClassifier()
        score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
        print(f"The default metrics for MPL model are: {score}")
    else:
        best_score = 0
        best_score_mean = 0
        best_param = None
        for r in range(20):
            parameters = choose_mlp_parameters()
            model = MLPClassifier(hidden_layer_sizes=parameters[0], max_iter=parameters[1],
                                  activation=parameters[2], learning_rate_init=parameters[3])
            score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
            mean = statistics.mean(np.array(score))
            if mean > best_score_mean:
                best_score = score
                best_score_mean = mean
                best_param = parameters
        print(f"The estimated optimal parameters are: {best_param}")
        print(f"Those parameters return the following metrics: {best_score}")

else:
    """
    Instantiate the decision tree.
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
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
        for r in range(20):
            parameters = choose_mlp_parameters()
            model = DecisionTreeClassifier(criterion=parameters[0], max_depth=parameters[1])

            score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
            mean = statistics.mean(np.array(score))
            if mean > best_score_mean:
                best_score = score
                best_score_mean = mean
                best_param = parameters
        print(f"The estimated optimal parameters are: {best_param}")
        print(f"Those parameters return the following metrics: {best_score}")
