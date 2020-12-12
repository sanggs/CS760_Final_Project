import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# label is the 19th column in the dataset
label_index = 19


def get_dataset(filename_list):
    y = np.array([])
    X = None
    for filename in filename_list:
        print("Adding %s to Training data" % filename)
        data = np.genfromtxt(filename, delimiter=",", skip_header=1)
        num_samples = data.shape[0]
        num_features = data.shape[1] - 1
        print("Number of features: {} \nNumber of samples: {}\n".format(num_features, num_samples))
        y = np.append(y, data[:, label_index])
        new_X = np.zeros((num_samples, num_features))
        new_X[:, 0:label_index] = data[:, 0:label_index]
        new_X[:, label_index:] = data[:, label_index + 1:]
        if X is None:
            X = new_X
        else:
            X = np.append(X, new_X, axis=0)

    print("Total number of training samples: %d" % len(X))
    return X, y


def get_test_data(test_file):
    data = np.genfromtxt(test_file, delimiter=",", skip_header=1)
    num_samples = data.shape[0]
    num_features = data.shape[1] - 1
    print("\nRunning trained model on %s data containing %d samples" % (test_file, num_samples))
    y_test = data[:, label_index]
    X_test = np.zeros((num_samples, num_features))
    X_test[:, 0:label_index] = data[:, 0:label_index]
    X_test[:, label_index:] = data[:, label_index + 1:]

    return X_test, y_test


def print_cross_validataion(cross_validation_accuracy, scoring):
    print("\nCross validation training results:")
    for k in scoring:
        key = 'train_' + k
        print(
            "%s: %f (+/- %f)" % (key, cross_validation_accuracy[key].mean(), cross_validation_accuracy[key].std() * 2))
        key = 'test_' + k
        print(
            "%s: %f (+/- %f)" % (key, cross_validation_accuracy[key].mean(), cross_validation_accuracy[key].std() * 2))


def print_test_scores(y_true, y_pred):
    print("Accuracy: %f" % accuracy_score(y_true, y_pred))
    print("Precision: %f" % precision_score(y_true, y_pred))
    print("Recall: %f" % recall_score(y_true, y_pred))
    print("F1 Score: %f" % f1_score(y_true, y_pred))
