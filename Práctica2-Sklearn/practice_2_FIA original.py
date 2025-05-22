import itertools
import statistics
import numpy as np

from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import KFold
from ucimlrepo import fetch_ucirepo

"""
EXPERIMENT PARAMETERS
"""
NUMBER_OF_FOLDS = 10
FEATURES_TO_USE = ["studytime", "failures", "schoolsup", "health", "absences"]

"""
These will determine which parameter are used in both MLP and Decision Tree models.
Switching default to True will show a model with default parameters.
"""
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

def print_introduction(data):
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
          "iterations, might be insufficient and cause the model to be trained differently in each execution.\n")

"""
METRICS
Accuracy is not always the best metric to measure how good a trained model is, so we will get several metrics.
"""

def get_metrics(true_labels, pred_labels):
    TP = np.sum((true_labels == 1) & (pred_labels == 1))
    TN = np.sum((true_labels == 0) & (pred_labels == 0))
    FP = np.sum((true_labels == 0) & (pred_labels == 1))
    FN = np.sum((true_labels == 1) & (pred_labels == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return accuracy, recall, precision, f1

"""
Cross Validation
In order to evaluate a training approach, the division of data into training and test subsets can lead to erroneous 
conclusions due to chance. 
In order to make the conclusions more reliable, there is the Cross Validation technique. In this technique, several 
training and test divisions are created, training is performed, metrics are obtained for each of them and finally 
an average of the metrics obtained is made.
"""

@ignore_warnings(category=ConvergenceWarning)
def cross_validation(model, data, labels, K):

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
MODEL SEARCHING FUNCTION DEFINITIONS
In order to simplify the code, some methods are defined for their later use.

*CAUTION* --> If new parameters were to be added, build_model method MUST BE modified.
"""

def build_model(params):
    if TRAIN_MLP:
        return MLPClassifier(hidden_layer_sizes=params[0], max_iter=params[1],
                             activation=params[2], learning_rate_init=params[3])
    else:
        return DecisionTreeClassifier(criterion=params[0], max_depth=params[1])

def default_classifier(data, labels):
    def_params = ((100,), 200, 'relu', 0.001) if TRAIN_MLP else ('gini', None)
    def_model = build_model(def_params)
    def_score = cross_validation(def_model, data, labels, NUMBER_OF_FOLDS)
    model_name = "MLP" if TRAIN_MLP else "Decision Tree"
    print(f"\nThe default metrics for {model_name} model are: ")
    print(f"Accuracy={def_score[0]}, Recall={def_score[1]}, Precision={def_score[2]}, F1={def_score[3]}")

    def_score_mean = statistics.mean(np.asarray(def_score))

    return def_score, def_score_mean, def_params

def opt_model_search (def_score, data, labels):
    print("\nSearching for optimal models...")
    combinations = itertools.product(LAYERS, MAX_ITER, FUNC, LR) if TRAIN_MLP else itertools.product(CRIT, MAX_DEPTH)
    counter = 0

    best_score = def_score[0]
    best_score_mean = def_score[1]
    best_param = def_score[2]

    for parameters in tqdm(combinations, desc="Testing models"):
        counter += 1
        model = build_model(parameters)
        score = cross_validation(model, data, labels, NUMBER_OF_FOLDS)
        mean = statistics.mean(np.array(score))
        if mean > best_score_mean:
            print("\nNew best model found!")
            print(f"The model number {counter} has been considered optimal.\n")
            print("\nSearching for optimal models...")
            best_score = score
            best_score_mean = mean
            best_param = parameters
    print("Search complete.")
    print(f"\nThe estimated optimal parameters are: {best_param}")
    print("Those parameters return the following metrics: ")
    print(f"Accuracy={best_score[0]}, Recall={best_score[1]}, Precision={best_score[2]}, F1={best_score[3]}")

"""
LOAD DATA
Load dataset included in scikit learn. This is a binary dataset, so the label feature can only be 1 or 0.
It is loaded as a Pandas dataframe and with (data, labels) format.
"""
# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
data = student_performance.data.features
labels = student_performance.data.targets

"""
CLEAN DATA
Now we clean the data. Some data is converted to binary values in order to implement MLP correctly.
"""
data.loc[:,'schoolsup'] = data['schoolsup'].map(lambda x: 1 if x == 'yes' else 0)
labels = labels['G3'].map(lambda x: 0 if x < 10 else 1)

data = data[FEATURES_TO_USE].head(500)
labels = labels.head(500)

"""
MODEL SEARCHING
We search for the best model among every combination of the predefined parameters
"""
print_introduction(data)

def_score = default_classifier(data, labels)
opt_model_search(def_score, data, labels)