import models, introduction
import itertools

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

COMBINATIONS = itertools.product(LAYERS, MAX_ITER, FUNC, LR) if TRAIN_MLP else itertools.product(CRIT, MAX_DEPTH)

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
introduction.print_introduction(data)

def_score = models.default_classifier(TRAIN_MLP, data, labels, NUMBER_OF_FOLDS)
models.opt_model_search(TRAIN_MLP, def_score, data, labels, NUMBER_OF_FOLDS, COMBINATIONS)