from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

from torchsummary import summary

# Training hyperparameters
epochs = 1
batch_size = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 20
# Create dictionary of target classes
label_dict = {
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
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class InceptionA(nn.Module):
    def __init__(self,input_channels):
        super(InceptionA, self).__init__()
        # print("112 - ",input_channels)
        #Create branch 1 with 1x1 kernel
        self.branch1x1 = nn.Conv2d(input_channels,16, kernel_size=1)

        # Create branch 2 with 5x5 kernel
        self.branch5x5_1 = nn.Conv2d(input_channels, 16, kernel_size= 1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # Create branch 3
        self.branch3x3_1 = nn.Conv2d(input_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # Create branch 4
        self.branch_pool = nn.Conv2d(input_channels, 24, kernel_size=1)

    def forward(self, x):
        # print("forward 12 - ", x.size())
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24,
                               kernel_size=5)

        # create the second Convolution layer
        self.conv2 = nn.Conv2d(in_channels=88, out_channels=32,
                               kernel_size=5)
        self.conv2_dropout = nn.Dropout2d(p = 0.15)
        self.inception1 = InceptionA(input_channels = 24)
        self.inception2 = InceptionA(input_channels = 32)

        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = F.relu(x)
        #print(x.size)
        x = self.inception1(x)

        x = self.conv2(x)
        # x = self.conv2_dropout(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.inception2(x)

        x = x.view(input_size, -1)
        x = self.fc(x)
        return F.log_softmax(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.batchnorm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=5, stride=1, padding=0)

        # create the second Convolution layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=5, stride=1, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                                kernel_size=3, stride=1, padding=0)

        self.mPool = nn.MaxPool2d(2)
        self.nn_dropout = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(in_features=8192, out_features=512)
        
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128
                             , out_features=10)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
#        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        input_size = x.size(0)
        
        #Batch normalization
        x = self.batchnorm(x)
        
        #First CNN
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.shape)
        
        #Second CNN
        x = self.conv2(x)
        x = self.mPool(x)
        x = F.relu(x)
        # print(x.size())
        
        #Third CNN
        x = self.conv3(x)
        x = F.relu(x)
        x = self.nn_dropout(x) #drop 50 %
        
        #Getting ready for connecting to FC
        x = x.view(input_size, -1)

        #First FC layer
        x = self.fc1(x)
        x = F.relu(x)
        
        #Second FC layer
        x = self.fc2(x)
        x = F.relu(x)
        
        #Third FC layer
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def plot_data(data, label, text):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        # print(label_dict[int("{}".format(label[i]))])
        plt.title(text + ": {}".format(label[i]))
#        plt.title(label_dict[int("".format(label[i]))])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def predict_batch(model, device, test_loader):
    examples = enumerate(test_loader)
    model.eval()
    with torch.no_grad():
        batch_idx, (data, target) = next(examples)
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        pred = pred.numpy()
    return data, target, pred


def plot_graph(train_x, train_y, test_x, test_y, ylabel=''):
    fig = plt.figure()
    plt.plot(train_x, train_y, color='blue')
    plt.plot(test_x, test_y, color='red')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def train(model, device, train_loader, optimizer, epoch, losses=[], counter=[], errors=[]):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            losses.append(loss.item())
            counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    errors.append(100. * (1 - correct / len(train_loader.dataset)))


def test(model, device, test_loader, losses=[], errors=[]):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    losses.append(test_loss)
    errors.append(100. * (1 - correct / len(test_loader.dataset)))

def save_predictions(model, device, test_loader, path):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), dim=1)
            with open(path, "a") as out_file:
                np.savetxt(out_file, output)


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # data transformation
    train_data = datasets.FashionMNIST('data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    test_data = datasets.FashionMNIST('data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

    # data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

    # extract and plot random samples of data
    examples = enumerate(test_loader)
    batch_idx, (data, target) = next(examples)
    # plot_data(data, target, 'Truth')

    # model creation
    #model = FCN().to(device)
    model = CNN().to(device)
    
#    model.load_state_dict(torch.load('3c3f.pt'))
#    model = InceptionNet().to(device)
#    model = CNNModel().to(device)
    # optimizer creation
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.001)
#    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # lists for saving history
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]
    train_errors = []
    test_errors = []
    error_counter = [i * len(train_loader.dataset) for i in range(epochs)]

    # test of randomly initialized model
    test(model, device, test_loader, losses=test_losses)

#     global training and testing loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, losses=train_losses, counter=train_counter, errors=train_errors)
        test(model, device, test_loader, losses=test_losses, errors=test_errors)
#
#    # plotting training history
    plot_graph(train_counter, train_losses, test_counter, test_losses, ylabel='negative log likelihood loss')
    plot_graph(error_counter, train_errors, error_counter, test_errors, ylabel='error (%)')
#
#    # extract and plot random samples of data with predicted labels
    data, _, pred = predict_batch(model, device, test_loader)
    plot_data(data, pred, 'Predicted')
#    
#    #Save the model
    torch.save(model.state_dict(), '3c3f.pt')
#    
    save_predictions(model, device, test_loader, "E:\\predictions.txt")
    summary(model, (1,28,28))

if __name__ == '__main__':
    main()
