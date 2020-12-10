import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate

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

model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
scoring = ('accuracy', 'precision', 'recall', 'f1')
scaled_y = minmax_scale(y)
cross_validation_accuracy = cross_validate(model, X, scaled_y, cv=10, scoring=scoring, return_train_score=True)

for k in scoring:
    key = 'train_' + k
    print("%s: %f (+/- %f)" % (key, cross_validation_accuracy[key].mean(), cross_validation_accuracy[key].std() * 2))
    key = 'test_' + k
    print("%s: %f (+/- %f)" % (key, cross_validation_accuracy[key].mean(), cross_validation_accuracy[key].std() * 2))
