import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt

from constants import Hyperparams

training_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

testing_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

def preview_data(dataset):
    rows, cols = 3, 3
    fig = plt.figure(figsize=(16, 9))
    for i in range(1, rows*cols + 1):
        data_index = torch.randint(0, len(dataset), (1,)).item()
        sample, label = dataset[data_index]
        plt.subplot(rows, cols, i)
        plt.imshow(sample.squeeze(), cmap="gray")
        plt.title(label)

    fig.tight_layout()
    plt.show()

train_loader = DataLoader(training_data, batch_size=Hyperparams.BATCH_SIZE.value, shuffle=True)
test_loader = DataLoader(testing_data, batch_size=Hyperparams.BATCH_SIZE.value, shuffle=True)

if __name__ == "__main__":
    preview_data(training_data)