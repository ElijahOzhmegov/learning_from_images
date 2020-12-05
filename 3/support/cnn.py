# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CATEGORIES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


# implement your own NNs
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # TODO: YOUR CODE HERE

    def forward(self, x):
        # TODO: YOUR CODE HERE
        return x

    def name(self):
        return "MyNeuralNetwork"


def training(model, data_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # backward
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if batch_idx % 10 == 0:
            print(f'Training Batch: {batch_idx:4} of {len(data_loader)}')

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


def test(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # do not compute gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


def plot(train_history, test_history, metric, num_epochs):

    plt.title(f"Validation/Test {metric} vs. Number of Training Epochs")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Test {metric}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), test_history, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{metric}.png")
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seed for reproducability
torch.manual_seed(0)

# hyperparameter
# TODO: find good hyperparameters
# batch_size = ...
# num_epochs = ...
# learning_rate = ...
# momentum = ...

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

# load train and test data
root = './data'
train_set = datasets.FashionMNIST(root=root,
                                  train=True,
                                  transform=transform,
                                  download=True)
test_set = datasets.FashionMNIST(root=root,
                                 train=False,
                                 transform=transform,
                                 download=True)

loader_params = {
    'batch_size': batch_size,
    'num_workers': 0  # increase this value to use multiprocess data loading
}

train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_params)
test_loader = DataLoader(dataset=test_set, shuffle=False, **loader_params)

## model setup
model = MyNeuralNetwork().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

train_acc_history = []
test_acc_history = []

train_loss_history = []
test_loss_history = []

best_acc = 0.0
since = time.time()

for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # train
    training_loss, training_acc = training(model, train_loader, optimizer,
                                           criterion, device)
    train_loss_history.append(training_loss)
    train_acc_history.append(training_acc)

    # test
    test_loss, test_acc = test(model, test_loader, criterion, device)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    # overall best model
    if test_acc > best_acc:
        best_acc = test_acc
        #  best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print(
    f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'
)
print(f'Best val Acc: {best_acc:4f}')

# plot loss and accuracy curves
train_acc_history = [h.cpu().numpy() for h in train_acc_history]
test_acc_history = [h.cpu().numpy() for h in test_acc_history]

plot(train_acc_history, test_acc_history, 'accuracy', num_epochs)
plot(train_loss_history, test_loss_history, 'loss', num_epochs)

# plot examples
example_data, _ = next(iter(test_loader))
with torch.no_grad():
    output = model(example_data)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Pred: {}".format(CATEGORIES[output.data.max(
            1, keepdim=True)[1][i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("examples.png")
    plt.show()
