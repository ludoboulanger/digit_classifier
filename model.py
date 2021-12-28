import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt

from constants import Constants, Hyperparams

from data import train_loader, test_loader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DigitClassifier(nn.Module):
    def __init__(self) -> None:
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5) # out = 24x24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # out = 12x12
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2) #out 4x4
        self.fc1 = nn.Linear(in_features=(32 * 2 * 2), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train(data, model, optimizer, loss_function):
    size = len(data.dataset)
    for batch, (X, y) in enumerate(data):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(X)
        loss = loss_function(pred, y)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(data, model, loss_function):
    size = len(data.dataset)
    num_batches = len(data)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in data:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def show_predictions():
    cols, rows = 3, 3

    model = load_model()

    samples = next(iter(test_loader))
    images, labels = samples
    model_outputs = model(images[:int(cols*rows)])
    y_hats = torch.max(model_outputs, 1)[1].numpy()

    plt.figure()
    for i in range(len(y_hats)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label : {labels[i]} Prediction : {y_hats[i]}")
    plt.show()


def load_model():
    model = DigitClassifier()
    model.load_state_dict(torch.load(Hyperparams.MODEL_PATH.value))

    return model


if __name__ == '__main__':
    model = DigitClassifier().to(DEVICE)
    opt = Adam(model.parameters(), lr=Hyperparams.LEARNING_RATE.value)
    loss_func = CrossEntropyLoss()

    for t in range(Hyperparams.EPOCHS.value):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, opt, loss_func)
        test(test_loader, model, loss_func)

    torch.save(model.state_dict(), Constants.MODEL_PATH.value)

    print("Training Done!")


