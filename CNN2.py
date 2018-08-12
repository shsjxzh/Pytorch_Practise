import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Hyper Parameters
EPOCH = 1                       # the training times
BATCH_SIZE = 64                 # not use all data to train
LR = 0.001
DOWNLOAD_MNIST = False          # if have already download, then turn it to 'False'
SHOW_STEP = 100                 # show the result after how many steps


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),  # prevent overfitting
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % SHOW_STEP == SHOW_STEP - 1:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / SHOW_STEP))
            running_loss = 0.0

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]                           # get the index of max prob
            correct = pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: %.4f, Accuracy: ')




# transform: to tensor format and do batch normalization
my_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_data = torchvision.datasets.MNIST(
    root='./mnist',             # the location to save
    train=True,
    download=DOWNLOAD_MNIST,
    transform=my_transform
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    download=DOWNLOAD_MNIST,
    transform=my_transform
)

# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

print(test_data)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# total = test_loader.__len__()
# print(total)


cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

print('Finished Training')

if __name__ == '__main__':
    main()