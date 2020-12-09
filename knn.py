import argparse
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import plot, show

N_CROSS_VAL = 10

arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("--dataset", type=str, default='data/processed_data.csv')
arg_parser.add_argument("--dataset", type=str, default='april_processed_final_grouped.csv')
args = arg_parser.parse_args()

filename = args.dataset
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

# scaler = MinMaxScaler()
# scaler.fit(X)
# print(scaler.data_max_)
# print(scaler.data_min_)
# print(X[0])
# print(scaler.transform(X)[0])

scaler = MinMaxScaler()
knn = KNeighborsClassifier(n_neighbors=20)
kf = KFold(n_splits=N_CROSS_VAL, shuffle=True)

accuracy1 = np.array([])
accuracy2 = np.array([])
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracy1 = np.append(accuracy1, [acc])
    print("\n\nNext iter:")
    print(acc)
    print(confusion_matrix(y_test, y_pred))
    scaler.fit(X_train)
    knn.fit(scaler.transform(X_train), y_train)
    y_pred = knn.predict(scaler.transform(X_test))
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracy2 = np.append(accuracy2, [acc])
    print(acc)
    print(confusion_matrix(y_test, y_pred))


print("Accuracy = " + str(accuracy1.mean()))
print("Accuracy = " + str(accuracy2.mean()))
