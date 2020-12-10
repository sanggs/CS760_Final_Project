import argparse
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

N_CROSS_VAL = 10

arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("--dataset", type=str, default='data/processed_data.csv')
arg_parser.add_argument("--dataset", type=str, default='april_processed_final_grouped.csv')
arg_parser.add_argument("-k", type=int, default=20)
args = arg_parser.parse_args()

filename = args.dataset
knn_k = args.k
data = np.genfromtxt(filename, delimiter=",", skip_header=1)

num_samples = data.shape[0]
num_features = data.shape[1] - 1
print("Number of features: {} \nNumber of samples: {}".format(num_features, num_samples))

# label is the 19th column in the dataset
label_index = 19
y = data[:, label_index]
X = np.zeros((num_samples, num_features))
X[:, 0:label_index] = data[:, 0:label_index]
X[:, label_index:] = data[:, label_index + 1:]

knn = KNeighborsClassifier(n_neighbors=knn_k)
kf = KFold(n_splits=N_CROSS_VAL, shuffle=True)

accuracy = np.array([])
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracy = np.append(accuracy, [acc])
    # print(acc)
    # print(confusion_matrix(y_test, y_pred))
    # print()

print(accuracy)
print("Accuracy = " + str(accuracy.mean()))