import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse

# Hyper Parameters
EPOCH = 50                     # the training times
BATCH_SIZE = 32                 # not use all data to train
LR = 0.007
SHOW_STEP = 100                 # show the result after how many steps
CHANGE_EPOCH = 5

# Data Describe
num_people = 10177
pic_after_MaxPool = 512 * 4 * 4
ImageSize = 128

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1 ** (epoch // CHANGE_EPOCH))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c_epoch', type=int, default=0)
    args = parser.parse_args()

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
    # print('I am here')
    train_loader = Data.DataLoader(dataset=face_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) #)#,  
    # print("ok")

    from my_vgg19_b import my_vgg19_b
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if args.c_epoch > 0:
        pretrain_sign = True
    else:
        pretrain_sign = False

    model = my_vgg19_b(num_classes=num_people, pic_size=pic_after_MaxPool, pretrained=pretrain_sign)
    model.load_state_dict(torch.load('c_params.pkl'))
    model = nn.DataParallel(model, device_ids=[0, 1, 2]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.c_epoch, EPOCH):
        # if epoch == 0:
            # print('I am here')
        if epoch % CHANGE_EPOCH == CHANGE_EPOCH - 1:
            adjust_learning_rate(optimizer, epoch)
        train(model, device, train_loader, optimizer, loss_func, epoch)

        if epoch % CHANGE_EPOCH == CHANGE_EPOCH - 1:
            torch.save(model.state_dict(), 'c_params.pkl')

    print('Finished Training')
    torch.save(model.state_dict(), 'c_params.pkl')
    # save the model

if __name__ == '__main__':
    main()