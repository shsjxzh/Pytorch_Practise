# the size of the picture is 128 * 128
# all the construction of net is over
# all the net are trained on GPU

import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import torchvision.models as models

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Hyper Parameters
EPOCH = 50                     # the training times
BATCH_SIZE = 2                 # not use all data to train
SHOW_STEP = 100                # show the result after how many steps
CHANGE_EPOCH = 5
USE_GPU = False

IC_LR = 0.001
A_LR = 0.001
G_LR = 0.001
D_LR = 0.001

# Data Describe
num_people = 10177
pic_after_MaxPool = 512 * 4 * 4

# other parameters
ImageSize = 128
DeviceID = [0]

# tmp generator
class Generator(nn.Module):
    def __init__(self, nz = 512 * 4 * 4 * 2, ngf = 64, nc = 3):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        output = self.main(input)
        return output


# tmp classifier
class Classifier(nn.Module):
    def __init__(self, num_classes=10177, pic_size=512 * 4 * 4, hidden_node=4096):
        super(Classifier, self).__init__()
        self.features = models.vgg19_bn(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(pic_size, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, hidden_node),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_node, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output, x


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

    train_loader = Data.DataLoader(dataset=face_data, batch_size=BATCH_SIZE, shuffle=True)#) #,num_workers=2   

    if USE_GPU:
        device = torch.device("cuda:" + str(DeviceID[0]))
    else:
        device = torch.device("cpu")

    # from my_vgg19_b import my_vgg19_b
    # since I and C are the same, we only use one net
    # IC = my_vgg19_b(num_classes=num_people, pic_size=pic_after_MaxPool, pretrained=True)
    IC = Classifier()
    if USE_GPU:
        IC = nn.DataParallel(IC, device_ids=DeviceID).to(device)
    IC_optimizer = torch.optim.Adam(IC.parameters(), lr=IC_LR)

    from AttributeDecoder import AttributeDecoder_19_b
    # we will consider pretrained later
    A = AttributeDecoder_19_b(size_after_max_pool=pic_after_MaxPool, use_gpu=USE_GPU)# , pretrained=True)
    if USE_GPU:   
        A = nn.DataParallel(A, device_ids=DeviceID).to(device)
    A_optimizer = torch.optim.Adam(A.parameters(), lr=A_LR)

    # from my_Re_vgg19_b import my_Re_vgg19_b
    # G = my_Re_vgg19_b()
    # test code
    G = Generator()
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
        for batch_idx, (subject, identity) in enumerate(train_loader):
            if USE_GPU:   
                subject, identity = subject.to(device), identity.to(device)
            if batch_idx % 2 == 0:
                attribute = subject
                r = 1
            else:
                r = 0.1

            # LIC loss
            IC_output, IC_sub = IC(subject)
            LIC_loss = nn.CrossEntropyLoss()(IC_output, identity)
            # backward to save memory
            IC_optimizer.zero_grad()
            LIC_loss.backward(retain_graph = True)
            IC_optimizer.step()

            # LK loss
            A_output, my_mean, log_var = A(subject)
            LKL_loss = 0.5 * (my_mean.pow(2) + log_var.exp() - log_var - 1).sum()

            input_vector = torch.cat((IC_sub, A_output), 1)
            input_vector = input_vector.unsqueeze(2).unsqueeze(3)
            
            print(input_vector.size())
            g_image = G(input_vector)
            print(g_image.size())
            print(subject.size())

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

