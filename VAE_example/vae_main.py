import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data 
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# Hyper Parameters
EPOCH = 200                     # the training times
BATCH_SIZE = 64                 # not use all data to train
LR = 0.001
DOWNLOAD_MNIST = False          # if have already download, then turn it to 'False'
SHOW_STEP = 100                 # show the result after how many steps
use_gpu = True
DeviceID = [2]

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),   # 64 * 28 * 28
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128 * 28 * 28
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)                   # 128 * 14 * 14
        )                  
        self.log_var = nn.Sequential(nn.Linear(128 * 14 * 14, 1024))
        self.my_mean = nn.Sequential(nn.Linear(128 * 14 * 14, 1024))
        self.reconstract = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )
    
    def encode(self, x):
        x = self.feature_extract(x)
        x = x.view(x.size(0), -1)
        return self.my_mean(x), self.log_var(x) 

    def reparametrize(self, my_mean, log_var):
        std = (0.5 * log_var).exp()
        
        eps = torch.normal(torch.zeros_like(std), 1)
        if use_gpu:
            device = torch.device("cuda:" + str(my_mean.get_device()))
        else:
            device = torch.device("cpu")

        eps = eps.to(device)

        return eps * std + my_mean

    def decode(self, z):
        return self.reconstract(z)

    def forward(self, x):
        my_mean, log_var = self.encode(x)
        z = self.reparametrize(my_mean, log_var)
        return self.decode(z), my_mean, log_var

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

# transform: to tensor format and do batch normalization
my_transform = transforms.Compose([
    transforms.ToTensor(), 
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def main():
    if not os.path.exists('./vae_img'):
        os.mkdir('./vae_img')

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

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    if use_gpu:
        device = torch.device("cuda:" + str(DeviceID[0]))
    else:
        device = torch.device("cpu")

    model = VAE()
    if use_gpu:
        model = nn.DataParallel(model, device_ids=DeviceID).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # train
    model.train()
    running_loss = 0.0
    for epoch in range(EPOCH):
        for batch_idx, (image, identity) in enumerate(train_loader):
            image, identity = image.to(device), identity.to(device)
            recon_batch, my_mean, log_var = model(image)

            # reconstruction loss
            rcstrc_loss = nn.MSELoss(size_average=False)
            L_R = rcstrc_loss(recon_batch, image.view(image.size(0), -1))

            # KL Divergence loss
            L_KLD = 0.5 * (my_mean.pow(2) + log_var.exp() - log_var - 1).sum()

            loss = L_R + L_KLD

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % SHOW_STEP == SHOW_STEP - 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / SHOW_STEP))
                running_loss = 0.0

        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img/image_{}.png'.format(epoch + 1))
            torch.save(model.state_dict(), './vae_params.pkl')

if __name__ == '__main__':
    main()


