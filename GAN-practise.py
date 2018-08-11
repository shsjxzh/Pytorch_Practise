import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1) # reporducible
# np.random.seed(1)

# hyper-parameter
BATCH_SIZE = 128
LR_G = 0.0005
LR_D = 0.0005
N_IDEAS = 5             # noise
ART_COMPONENTS = 40     # the total points that can be drawn
PAINT_POINTS = np.vstack([np.linspace(-1,1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
MAX_STEP = 20001
DECAY_EPOCH = 3000

def artist_works():
    a = np.random.uniform(1,2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + a - 1
    paintings = torch.from_numpy(paintings).float()
    return paintings

def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


# Generator
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),        
)

# Discriminator
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid(),        
)

# optimizer
optim_G = torch.optim.Adam(G.parameters(), lr = LR_G)
optim_D = torch.optim.Adam(D.parameters(), lr = LR_D)

plt.ion()

for i in range(MAX_STEP):
    # adjust learning rate
    if i % DECAY_EPOCH == 0:
        LR_D = LR_D / 2
        LR_G = LR_G / 2
        adjust_learning_rate(optim_D, LR_D)
        adjust_learning_rate(optim_G, LR_G)


    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)
    G_paintings = G(G_ideas)
    
    prob_artist0 = D(artist_paintings)  # D try to increase this
    prob_artist1 = D(G_paintings)       # D try to reduce this 
    
    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = -torch.mean(torch.log(prob_artist1))
    
    if i % 3 == 0:
        optim_D.zero_grad()
        D_loss.backward(retain_graph = True) # reuse computational graph
        optim_D.step()
    
    optim_G.zero_grad()
    G_loss.backward()
    optim_G.step()
    
    if i % 100 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()