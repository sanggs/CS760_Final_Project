import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import minmax_scale
import os
from pathlib import Path

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
arg_parser.add_argument("--dataset", type=str, default='data/processed_data.csv')
arg_parser.add_argument("--epochs", type=int, default=100)
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
scaled_X = torch.tensor(minmax_scale(X), dtype=torch.float32)
scaled_y = torch.tensor(minmax_scale(y), dtype=torch.float32)

print("Splitting training and test data as 90-10")
num_test_samples = int(num_samples/10)
num_train_samples = num_samples - num_test_samples

train_data = scaled_X[0:num_train_samples, :]
train_labels = scaled_y[0:num_train_samples]

test_data = scaled_X[num_train_samples:, :]
test_labels = scaled_y[num_train_samples:]

model = FullyConnectedNN(num_features)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

training_batch_size = 100
num_epochs = args.epochs

train_dataset = TensorDataset(train_data,train_labels)
data_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)

print("Using loss as Binary Cross Entropy Loss")
loss = torch.nn.BCELoss()

loss_list = []

for e in range(num_epochs):
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
	print(total_loss)
	loss_list.append(total_loss.item())

print("Saving the trained model")
this_filepath = Path(os.path.abspath(__file__))
this_dirpath = this_filepath.parent

model_path = os.path.join(this_dirpath, "model")
if not (os.path.exists(model_path)):
	os.makedirs(model_path)
model_path = os.path.join(model_path, "fnn.log")
torch.save(model, model_path)

import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.show()

print("Evaluating the trained model now:")

def get_predictions_from_model_output(model_output):
	zeros = torch.zeros(model_output.shape)
	ones = torch.ones(model_output.shape)
	model_prediction = torch.where(model_output >= 0.5, ones, zeros)
	return model_prediction

model = torch.load(model_path)
model.eval()

model_input = test_data
model_output = model(model_input)
model_prediction = get_predictions_from_model_output(model_output)

true_label = test_labels
accuracy = float(torch.sum(true_label == model_prediction.view(-1)).item())/float(true_label.shape[0])

from sklearn.metrics import classification_report
target_names = ['presence of mental health condition', 'absence of mental health condtion']
report = classification_report(true_label.view(-1).numpy(), model_prediction.view(-1).numpy(), target_names=target_names)

print("-----------Classification report of the FullyConnected Neural Network-----------")
print(report)

