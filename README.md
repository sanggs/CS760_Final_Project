# CS760 Final Project
Authors: Sangeetha Grama Srinivasan, Vinith Venkatesan

In this project, we aim to implement classifiers that identify if an individual has a mental health condition or not, based on the information about the individual obtained from a survey related to COVID-19. The processed dataset has been put under **data/** folder

### Dependencies: Python Libraries 
We assume that you have Python3 installed. To run the scripts, we require the following python libraries:
1. scikit-learn
2. numpy
3. os
4. argparse
5. PyTorch
6. matplotlib
7. tqdm
8. pathlib

## Steps to run the scripts in this repository:

### 1. Clone the repository and change the directory into the cloned folder: 
```
git clone git@github.com:sanggs/CS760_Final_Project.git
cd CS760_Final_Project/
```
### 2. Command to run every script
You can use this general command to run every script. Replace <script_name> with the name of the model you want to train, <file1> with data/april_processed_data.csv, <file2> with data/may_processed_data.csv and <file3> with data/june_processed_data.csv
```
python <script_name>.py --dataset <file1> --dataset <file2> --test <test_file>
```
Example: To run decision_tree.py, use 
```
python decision_tree.py --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
NOTE: The --test argument is not used for running knn_best_k.py. To see commands to run each script, please see the section below.

### 3. Commands for every script with all the options is given below
- knn.py
```
python knn.py -k 20 --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
 -k sets the value of k in KNN 
- knn_best_k.py
```
python knn_best_k.py --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv
```
This script can be used to determine accuracy for different values of k.
- decision_tree.py
```
python decision_tree.py --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
- random_forest.py
```
python random_forest.py --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
- naive_bayes.py
```
python naive_bayes.py --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
- logistic_regression.py
```
python logistic_regression.py --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
- svm.py
```
python svm.py --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
- neural_net.py
```
python neural_net.py --save-model --batch-size 150 --epochs 150 --dataset data/april_processed_data.csv --dataset data/may_processed_data.csv --test data/june_processed_data.csv
```
 --save-model is to save the trained neural network to a file,
 --batch-size sets the batch size for training the neural network,
 --epochs sets the number of epochs to train the neural network.
