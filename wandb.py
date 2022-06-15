import torch
import os
import wandb
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()

config=dict(
    in_channels = 1,
    num_classes = 10,
    learning_rate = 0.001,
    batch_size = 64,
    num_epochs = 3,
    slice=10,
    load_model = True,
    checkpoint_name = os.path.join("./checkpoints", "cnn_checkpoint.pth.tar" )
)

def save_checkpoint(state, config):
    print("=> Saving Checkpoint")
    torch.save(state, config.checkpoint_name)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def model_pipeline(hyperparameters):
    with wandb.init(project="cnn_demo",name = "pytorch", config=hyperparameters):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)
        train(model, train_loader, criterion, optimizer, config)
        print(f"Train Accuracy = {test(model, train_loader)}")
        print(f"Test Accuracy = {test(model, test_loader)}")
        
    return model

def make(config):
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

#     train_subset = torch.utils.data.DataLoader(train_dataset, indices=range(0, len(train_dataset), config.slice))
#     test_subset = torch.utils.data.DataLoader(test_dataset, indices=range(0, len(test_dataset), config.slice))

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=True)

    model = CNN(config.in_channels, config.num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer

def train(model, loader, criterion, optimizer, config):
    wandb.watch(model, criterion, log="all", log_freq=30)

    if config.load_model:
        load_checkpoint(torch.load(config.checkpoint_name), model, optimizer)
    
    for epoch in range(config.num_epochs):
        print(f"Training epoch: {epoch}")
        losses = []

        if epoch %2 == 0:
            checkpoint = {"model" : model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)
        example_ct = 0
        for batch_idx, (data, targets) in enumerate(tqdm(loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            example_ct += len(data)

            if batch_idx % 25 == 0:
                train_log(loss, example_ct, epoch)
    
def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after" + str(example_ct).zfill(5) + f"examples: {loss:.3f}")

def test(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:    # x=images, y=labels
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = (100*num_correct)/num_samples
        wandb.log({"Accuracy": accuracy})

    model.train()
    torch.onnx.export(model, x, "model.onnx")
    wandb.save("model.onnx")
    return accuracy

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    model = model_pipeline(config)