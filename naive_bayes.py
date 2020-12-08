import numpy as np
import argparse
import sklearn
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dataset", type=str, default='data/processed_data.csv')
args = arg_parser.parse_args()

filename = args.dataset
data = np.genfromtxt(filename, delimiter=",", skip_header=1)

num_samples = data.shape[0]
num_features = data.shape[1]-1
print("Number of features: {} \nNumber of samples: {}".format(num_features, num_samples))

# label is the 19th column in the dataset
label_index = 19
y = data[:, label_index]
X = np.zeros((num_samples, num_features))
X[:, 0:label_index] = data[:, 0:label_index]
X[:, label_index:] = data[:, label_index+1:]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

num_training_samples = X_train.shape[0]
num_training_samples = X_test.shape[0]

model = MultinomialNB().fit(X_train, y_train)
predicted = model.predict(X_test)
print(np.mean(predicted == y_test))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predicted))
