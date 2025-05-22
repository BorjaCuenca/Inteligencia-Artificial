import statistics
import cross_validation
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

def build_model(isMLP, params):
    if isMLP:
        return MLPClassifier(hidden_layer_sizes=params[0], max_iter=params[1],
                             activation=params[2], learning_rate_init=params[3])
    else:
        return DecisionTreeClassifier(criterion=params[0], max_depth=params[1])

def default_classifier(isMLP, data, labels, folds):
    def_params = ((100,), 200, 'relu', 0.001) if isMLP else ('gini', None)
    def_model = build_model(isMLP, def_params)
    def_score = cross_validation.get_scores(def_model, data, labels, folds)
    model_name = "MLP" if isMLP else "Decision Tree"
    print(f"\nThe default metrics for {model_name} model are: ")
    print(f"Accuracy={def_score[0]}, Recall={def_score[1]}, Precision={def_score[2]}, F1={def_score[3]}")

    def_score_mean = statistics.mean(np.asarray(def_score))

    return def_score, def_score_mean, def_params

def opt_model_search (isMLP, def_score_tuple, data, labels, folds, combinations):
    print("\nSearching for optimal models...")
    counter = 0

    best_score = def_score_tuple[0]
    best_score_mean = def_score_tuple[1]
    best_param = def_score_tuple[2]

    for parameters in tqdm(combinations, desc="Testing models"):
        counter += 1
        model = build_model(isMLP, parameters)
        score = cross_validation.get_scores(model, data, labels, folds)
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