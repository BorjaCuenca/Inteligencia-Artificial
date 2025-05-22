import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.utils._testing import ignore_warnings


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

@ignore_warnings(category=ConvergenceWarning)
def get_scores(model, data, labels, K):

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