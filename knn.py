import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-k", type=int, default=20)
arg_parser.add_argument("--dataset", action='append', default=[])
arg_parser.add_argument("--test", type=str, default='june_processed_final_grouped.csv')
args = arg_parser.parse_args()

knn_k = args.k
filename_list = args.dataset
test_file = args.test

# Default value when no command line args passed
if not filename_list:
    filename_list = ['april_processed_final_grouped.csv', 'may_processed_final_grouped.csv']

# label is the 19th column in the dataset
label_index = 19

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
print("\nCross validation training results:")

model = KNeighborsClassifier(n_neighbors=knn_k)
scoring = ('accuracy', 'precision', 'recall', 'f1')
scaled_y = minmax_scale(y)
cross_validation_accuracy = cross_validate(model, X, scaled_y, cv=10, scoring=scoring, return_train_score=True)

for k in scoring:
    key = 'train_' + k
    print("%s: %f (+/- %f)" % (key, cross_validation_accuracy[key].mean(), cross_validation_accuracy[key].std() * 2))
    key = 'test_' + k
    print("%s: %f (+/- %f)" % (key, cross_validation_accuracy[key].mean(), cross_validation_accuracy[key].std() * 2))

data = np.genfromtxt(test_file, delimiter=",", skip_header=1)
num_samples = data.shape[0]
num_features = data.shape[1] - 1
print("\nRunning trained model on %s data containing %d samples" % (test_file, num_samples))
y_test = data[:, label_index]
X_test = np.zeros((num_samples, num_features))
X_test[:, 0:label_index] = data[:, 0:label_index]
X_test[:, label_index:] = data[:, label_index + 1:]

model.fit(X, scaled_y)
y_test_scaled = minmax_scale(y_test)
y_pred = model.predict(X_test)
print("Accuracy: %f" % accuracy_score(y_test_scaled, y_pred))
print("Precision: %f" % precision_score(y_test_scaled, y_pred))
print("Recall: %f" % recall_score(y_test_scaled, y_pred))
print("F1 Score: %f" % f1_score(y_test_scaled, y_pred))
