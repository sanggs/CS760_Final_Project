import numpy as np
import argparse
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate

from utils import *

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dataset", action='append', default=[])
arg_parser.add_argument("--test", type=str, default='data/june_processed_data.csv')
args = arg_parser.parse_args()

filename_list = args.dataset
test_file = args.test

# Default value when no command line args passed
if not filename_list:
    filename_list = ['data/april_processed_data.csv', 'data/may_processed_data.csv']

X, y = get_dataset(filename_list)

scoring = ('accuracy','precision','recall','f1')

scaled_X = minmax_scale(X)
scaled_y = minmax_scale(y)
model = MultinomialNB()
cross_validation_accuracy = cross_validate(model, scaled_X, scaled_y, cv=10, scoring=scoring, return_train_score=True)
print_cross_validataion(cross_validation_accuracy, scoring)

X_test, y_test = get_test_data(test_file)

model.fit(scaled_X, scaled_y)
X_test_scaled = minmax_scale(X_test)
y_test_scaled = minmax_scale(y_test)
y_pred = model.predict(X_test_scaled)
print_test_scores(y_test_scaled, y_pred)

