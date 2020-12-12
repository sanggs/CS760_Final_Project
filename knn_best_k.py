import argparse
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import plot, xlabel, ylabel, show

from utils import get_dataset

N_CROSS_VAL = 10
KNN_K = 101

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dataset", action='append', default=[])
args = arg_parser.parse_args()

filename_list = args.dataset

# Default value when no command line args passed
if not filename_list:
    filename_list = ['data/april_processed_data.csv', 'data/may_processed_data.csv']

X, y = get_dataset(filename_list)

accuracies = np.array([])
for k in range(1, KNN_K):
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(knn, X, y, cv=N_CROSS_VAL)
    print("K = " + str(k) + ", Accuracy:" + str(accuracy.mean()))
    accuracies = np.append(accuracies, [accuracy.mean()])

print("Max Accuracy K value is " + str(np.argmax(accuracies)) + ", Accuracy = " + str(accuracies.max()))
plot(list(range(1, KNN_K)), list(accuracies))
xlabel("Value for K")
ylabel("Accuracy")
show()
