import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
from torchvision.utils import save_image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


NOISE_DIM = 96
BATCH_SIZE = 128

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!

def to_img(x):
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

mnist_train = dset.MNIST('./mnist', train=True, download=False,
                           # transform=T.ToTensor())
                            transform = T.Compose([
                                T.ToTensor(), 
                                T.Normalize(mean=[0.5], std=[0.5])
                            ]))

loader_train = DataLoader(mnist_train, batch_size=BATCH_SIZE)

def sample_noise(batch_size, dim):
    
    # torch.rand produces random values between 0 and 1 so we need to scale and shift.
    random_noise = torch.rand(batch_size, dim)*2 - 1

    return random_noise

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)

def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()
    
def discriminator_loss(logits_real, logits_fake):
    # Batch size.
    N = logits_real.size()
    
    # Target label vector, the discriminator should be aiming
    true_labels = Variable(torch.ones(N)).type(dtype)
    
    # Discriminator loss has 2 parts: how well it classifies real images and how well it
    # classifies fake images.
    real_image_loss = bce_loss(logits_real, true_labels)
    fake_image_loss = bce_loss(logits_fake, 1 - true_labels)
    
    loss = real_image_loss + fake_image_loss
    
    return loss

def generator_loss(logits_fake):
    # Batch size.
    N = logits_fake.size()
    
    # Generator is trying to make the discriminator output 1 for all its images.
    # So we create a 'target' label vector of ones for computing generator loss.
    true_labels = Variable(torch.ones(N)).type(dtype)
    
    # Compute the generator loss compraing 
    loss = bce_loss(logits_fake, true_labels)
    
    return loss
    
def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    return optimizer
    
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=100, 
              batch_size=128, noise_size=96, num_epochs=10):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            # real_data = Variable(x).type(dtype)
            # logits_real = D(2* (real_data - 0.5)).type(dtype)
            logits_real = x.type(dtype)  

            g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.data.item(),g_error.data.item()))
                save = to_img(fake_images.data.cpu().data)
                save_image(save, './gan_img/image_{}.png'.format(iter_count))
                # torch.save(D.state_dict(), './D.pkl')
                # torch.save(G.state_dict(), './G.pkl')
            iter_count += 1
            
def build_dc_classifier():
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        Unflatten(BATCH_SIZE, 1, 28, 28),
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
        nn.LeakyReLU(inplace=True, negative_slope=0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
        nn.LeakyReLU(inplace=True, negative_slope=0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(4*4*64, 4*4*64),
        nn.LeakyReLU(inplace=True, negative_slope=0.01),
        nn.Linear(4*4*64, 1),
    )

data = Variable(loader_train.__iter__().next()[0]).type(dtype)
b = build_dc_classifier().type(dtype)
out = b(data)

def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(1024, 7*7*128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=7*7*128),
        Unflatten(BATCH_SIZE, 128, 7, 7),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        Flatten(),
    )

test_g_gan = build_dc_generator().type(dtype)
test_g_gan.apply(initialize_weights)

fake_seed = Variable(torch.randn(BATCH_SIZE, NOISE_DIM)).type(dtype)
fake_images = test_g_gan.forward(fake_seed)

D_DC = build_dc_classifier().type(dtype) 
# D_DC.apply(initialize_weights)
G_DC = build_dc_generator().type(dtype)
# G_DC.apply(initialize_weights)

D_DC_solver = get_optimizer(D_DC)
G_DC_solver = get_optimizer(G_DC)

run_a_gan(D_DC, G_DC, D_DC_solver, G_DC_solver, discriminator_loss, generator_loss, num_epochs=5)