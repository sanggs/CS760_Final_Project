import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import minmax_scale
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from utils import *

class FullyConnectedNN(nn.Module):
	def __init__(self, input_size):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(input_size, 16),
			nn.ReLU(),
			nn.Linear(16, 4),
			nn.ReLU(),
			nn.Linear(4, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.model(x)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dataset", action='append', default=[])
arg_parser.add_argument("--test", type=str, default='data/june_processed_data.csv')
arg_parser.add_argument("--epochs", type=int, default=150)
arg_parser.add_argument("--batch-size", type=int, default=100)
arg_parser.add_argument("--save-model", action='store_true')
args = arg_parser.parse_args()

filename_list = args.dataset
test_file = args.test

# Default value when no command line args passed
if not filename_list:
    filename_list = ['data/april_processed_data.csv', 'data/may_processed_data.csv']

X, y = get_dataset(filename_list)

num_features = X.shape[1]

scaled_X = torch.tensor(minmax_scale(X), dtype=torch.float32)
scaled_y = torch.tensor(minmax_scale(y), dtype=torch.float32)

train_dataset = TensorDataset(scaled_X, scaled_y)

model = FullyConnectedNN(num_features)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

training_batch_size = args.batch_size
num_epochs = args.epochs

data_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)

print("Using loss as Binary Cross Entropy Loss")
loss = torch.nn.BCELoss()

loss_list = []

for e in tqdm(range(num_epochs)):
	total_loss = 0.0
	for batch in data_loader:
		optimizer.zero_grad()
		model_input = batch[0]
		label = batch[1].view(-1, 1)
		model_output = model(model_input)
		l = loss(model_output, label)
		l.backward()
		optimizer.step()
		total_loss += l
	tqdm.write("loss: " + str(total_loss.item()))
	loss_list.append(total_loss.item())

if (args.save_model):
	from pathlib import Path
	print("Saving the trained model")
	this_filepath = Path(os.path.abspath(__file__))
	this_dirpath = this_filepath.parent

	model_path = os.path.join(this_dirpath, "model")
	if not (os.path.exists(model_path)):
		os.makedirs(model_path)
	model_path = os.path.join(model_path, "fnn.log")
	torch.save(model, model_path)

plt.plot(loss_list)
plt.show()

print("Evaluating the trained model now:")

X_test, y_test = get_test_data(test_file)
X_test_scaled = torch.tensor(minmax_scale(X_test), dtype=torch.float32)
y_test_scaled = torch.tensor(minmax_scale(y_test), dtype=torch.float32)

def get_predictions_from_model_output(model_output):
	zeros = torch.zeros(model_output.shape)
	ones = torch.ones(model_output.shape)
	model_prediction = torch.where(model_output >= 0.5, ones, zeros)
	return model_prediction

def print_metrics(true_label, model_prediction):
	from sklearn import metrics
	accuracy = float(torch.sum(true_label == model_prediction.view(-1)).item())/float(true_label.shape[0])
	print("Accuracy: %f" % accuracy)
	print("Precision: %f" % metrics.precision_score(true_label.view(-1).numpy(), model_prediction.view(-1).numpy()))
	print("Recall: %f" % metrics.recall_score(true_label.view(-1).numpy(), model_prediction.view(-1).numpy()))
	print("F1 score: %f" % metrics.f1_score(true_label.view(-1).numpy(), model_prediction.view(-1).numpy()))

model.eval()

print("Getting metrics for training data")
model_input = scaled_X
model_output = model(model_input)
model_prediction = get_predictions_from_model_output(model_output)

true_label = scaled_y

target_names = ['presence of mental health condition', 'absence of mental health condtion']
report = classification_report(true_label.view(-1).numpy(), model_prediction.view(-1).numpy(), target_names=target_names)

print_metrics(true_label, model_prediction)

print("-----------Classification report of the FullyConnected Neural Network using Training data-------------")
print(report)

print("Getting metrics for testing data")
model_input = X_test_scaled
model_output = model(model_input)
model_prediction = get_predictions_from_model_output(model_output)

true_label = y_test_scaled

target_names = ['presence of mental health condition', 'absence of mental health condtion']
report = classification_report(true_label.view(-1).numpy(), model_prediction.view(-1).numpy(), target_names=target_names)

print_metrics(true_label, model_prediction)
print("-----------Classification report of the FullyConnected Neural Network using Testing data-------------")
print(report)
