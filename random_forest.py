import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate

from utils import *

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-k", type=int, default=20)
arg_parser.add_argument("--dataset", action='append', default=[])
arg_parser.add_argument("--test", type=str, default='data/june_processed_data.csv')
args = arg_parser.parse_args()

knn_k = args.k
filename_list = args.dataset
test_file = args.test

# Default value when no command line args passed
if not filename_list:
    filename_list = ['data/april_processed_data.csv', 'data/may_processed_data.csv']

X, y = get_dataset(filename_list)

model = RandomForestClassifier(criterion="entropy")
scoring = ('accuracy', 'precision', 'recall', 'f1')
scaled_y = minmax_scale(y)
cross_validation_accuracy = cross_validate(model, X, scaled_y, cv=10, scoring=scoring, return_train_score=True)
print_cross_validataion(cross_validation_accuracy, scoring)

X_test, y_test = get_test_data(test_file)

model.fit(X, scaled_y)
y_test_scaled = minmax_scale(y_test)
y_pred = model.predict(X_test)
print_test_scores(y_test_scaled, y_pred)
