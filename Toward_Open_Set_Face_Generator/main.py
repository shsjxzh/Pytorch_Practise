# the size of the picture is 128 * 128
# all the construction of net is over
# all the net are trained on GPU

import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import torchvision.models as models

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from torchvision import datasets, transforms

# Hyper Parameters
EPOCH = 100                    # the training times
BATCH_SIZE = 2                 # not use all data to train
SHOW_STEP = 100                # show the result after how many steps
CHANGE_EPOCH = 30
SAVE_EPOCH = 10
USE_GPU = True                # CHANGE THIS ON GPU!!

IC_LR = 0.0001
A_LR = 0.0001
G_LR = 0.0001
D_LR = 0.0001

# Data Describe
num_people = 10177
pic_after_MaxPool = 512 * 4 * 4

# other parameters
# be care about the image size !!
# ImageSize = 128
ImageSize = 224
VectorLength = 4096
DeviceID = [0]                  # CHANGE THIS ON GPU!!

# This will only be used on CPU!!
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = np.transpose(inp.detach().numpy(), (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def save_img(inp, filename):
    inp = np.transpose(inp.detach().numpy(), (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    mpimg.imsave(filename, inp)


def adjust_learning_rate(LR, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1 ** (epoch // CHANGE_EPOCH))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # load the data
    from CelebADataset import CelebADataset
    mytransform = transforms.Compose([
                    transforms.Resize(ImageSize),
                    transforms.CenterCrop(ImageSize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                    ])
    face_data = CelebADataset(csv_file='celeba_label/identity_CelebA.txt', 
                              root_dir='img_align_celeba',
                              transform=mytransform)

    # CHANGE THIS ON GPU!!
    train_loader = Data.DataLoader(dataset=face_data, batch_size=BATCH_SIZE, shuffle=True) #,num_workers=2)   

    if USE_GPU:
        device = torch.device("cuda:" + str(DeviceID[0]))
    else:
        device = torch.device("cpu")

    # from my_vgg19_b import my_vgg19_b
    # since I and C are the same, we only use one net
    # IC = my_vgg19_b(num_classes=num_people, pic_size=pic_after_MaxPool, pretrained=True)
    from Classifier import Classifier
    IC = Classifier(num_classes=num_people, vector_length=VectorLength)
    if USE_GPU:
        IC = nn.DataParallel(IC, device_ids=DeviceID).to(device)
    IC_optimizer = torch.optim.Adam(IC.parameters(), lr=IC_LR)

    # we will consider pretrained later
    from AttributeDecoder import AttributeDecoder
    A = AttributeDecoder(use_gpu=USE_GPU, vector_length=VectorLength)# , pretrained=True)
    if USE_GPU:   
        A = nn.DataParallel(A, device_ids=DeviceID).to(device)
    A_optimizer = torch.optim.Adam(A.parameters(), lr=A_LR)

    # from my_Re_vgg19_b import my_Re_vgg19_b
    # G = my_Re_vgg19_b()
    from Generator import Generator
    G = Generator(input_size=VectorLength * 2)
    if USE_GPU:
        G = nn.DataParallel(G, device_ids=DeviceID).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=G_LR)

    from Discriminator import Discriminator
    D = Discriminator()
    if USE_GPU:
        D = nn.DataParallel(D, device_ids=DeviceID).to(device)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=D_LR)
    
    # training
    fake_label = torch.full((BATCH_SIZE,), 0, device=device)
    
    for epoch in range(EPOCH):
        if epoch % CHANGE_EPOCH == CHANGE_EPOCH - 1:
            adjust_learning_rate(IC_LR, IC_optimizer, epoch)
            adjust_learning_rate(A_LR, A_optimizer, epoch)
            adjust_learning_rate(G_LR, G_optimizer, epoch)
            adjust_learning_rate(D_LR, D_optimizer, epoch)

        # train
        r = 0
        running_loss = 0
        attribute = 0
        for batch_idx, (subject, identity) in enumerate(train_loader):
            if USE_GPU:   
                subject, identity = subject.to(device), identity.to(device)
            if batch_idx % 2 == 0:
                attribute = subject
                r = 1
            else:
                r = 0.1

            # LIC loss
            IC_result, IC_sub = IC(subject)
            LIC_loss = nn.CrossEntropyLoss()(IC_result, identity)
            # backward to save memory
            IC_optimizer.zero_grad()
            LIC_loss.backward(retain_graph = True)
            IC_optimizer.step()

            # LK loss
            A_output, my_mean, log_var = A(attribute)
            LKL_loss = 0.5 * (my_mean.pow(2) + log_var.exp() - log_var - 1).sum()

            input_vector = torch.cat((IC_sub, A_output), 1)
            input_vector = input_vector.unsqueeze(2).unsqueeze(3)
            
            # print(input_vector.size())
            g_image = G(input_vector)
            # print(g_image.size())
            # print(subject.size())

            # if not USE_GPU:
              # imshow(torchvision.utils.make_grid(attribute))

            # LGD loss
            prob_sub, fd_image = D(subject)             # D try to increase this
            prob_gen, fd_g_image = D(g_image)           # D try to reduce this 
            LGD_loss = 0.5 * ((fd_g_image - fd_image)**2).sum()

            # LGR loss
            LGR_loss = 0.5 * ((g_image - subject)**2).sum()

            # LGC loss
            _, IC_atr = IC(attribute)
            LGC_loss = 0.5 * ((IC_atr - IC_sub)**2).sum()

            # LD loss
            # LD_loss = -torch.mean(torch.log(prob_sub) + torch.log(1. - prob_gen))
            LD_loss = nn.BCEWithLogitsLoss()(prob_gen, fake_label)

            D_optimizer.zero_grad()
            # LD loss: train in division
            # L_errD_fake.backward()
            # L_errD_real.backward()
            LD_loss.backward(retain_graph = True)
            D_optimizer.step()

            A_optimizer.zero_grad()
            (LKL_loss + r * LGR_loss).backward(retain_graph = True)
            A_optimizer.step()

            G_optimizer.zero_grad()
            (r * LGR_loss + LGD_loss + LGC_loss).backward()
            G_optimizer.step()

            # show the total error
            with torch.no_grad():
                running_loss += LIC_loss.item() + LD_loss.item() + (r * LGR_loss + LGD_loss + LGC_loss).item() + (LKL_loss + r * LGR_loss).item()
                if batch_idx % SHOW_STEP == SHOW_STEP - 1:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / SHOW_STEP))
                    running_loss = 0.0
        
        
        # if epoch % CHANGE_EPOCH == CHANGE_EPOCH - 1:
          #  torch.save(model.state_dict(), 'c_params.pkl')
        if epoch % SAVE_EPOCH == CHANGE_EPOCH - 1:
            save = torchvision.utils.make_grid(g_image)
            save_img(save, './img/g_image_{}.png'.format(epoch + 1))
            save = torchvision.utils.make_grid(subject)
            save_img(save, './img/identity_{}.png'.format(epoch + 1))
            save = torchvision.utils.make_grid(attribute)
            save_img(save, './img/attribute_{}.png'.format(epoch + 1))


    print('Finished Training')
    # torch.save(model.state_dict(), 'c_params.pkl')
    # save the model

if __name__ == '__main__':
    main()



# real_label = 1
# fake_label = 0

# label = torch.full((BATCH_SIZE,), real_label, device=device)
# L_errD_real = nn.BCELoss(fd_image, label)

# label.fill_(fake_label)
# L_errD_fake = nn.BCELoss(fd_g_image, label)

# with torch.no_grad():
    # make sure other loss all use original python type!
    # LD_loss = L_errD_fake + L_errD_real

