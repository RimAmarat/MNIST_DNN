import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tabulate import tabulate

# Load data
training = datasets.MNIST(root='./data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))
testing = datasets.MNIST(root='./data', train=False, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)

# Define the network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(784, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        data = F.relu(self.hidden2(data))
        data = self.output_layer(data)
        return F.log_softmax(data, dim=1)

network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.009)
epochs = 10

# Training loop
loss_table = [["Iteration", "Loss"]]
for i in range(epochs):
    for data in train_set:
        image, output = data
        network.zero_grad()
        result = network(image.view(-1, 784))
        loss = F.nll_loss(result, output)
        loss.backward()
        optimizer.step()
    loss_table.append([i, loss.item()])
    print("Iteration", i, "->", loss.item())

print(tabulate(loss_table, headers="firstrow", tablefmt="fancy_grid"))

# Evaluation
network.eval()
correct_pred = 0
incorrect_pred = 0
total = 0

with torch.no_grad():
    for data in test_set:
        image, output = data
        result = network(image.view(-1, 784))
        for index, tensor_val in enumerate(result):
            total += 1
            if torch.argmax(tensor_val) == output[index]:
                correct_pred += 1
            else:
                incorrect_pred += 1

accuracy = correct_pred / total
print("Accuracy:", accuracy)
