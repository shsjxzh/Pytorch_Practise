import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision # this is the database of torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Hyper Parameters
EPOCH = 500                     # the training times
BATCH_SIZE = 32                # not use all data to train
LR = 0.01
SHOW_STEP = 100                 # show the result after how many steps

# Data Describe
num_people = 10177
pic_after_MaxPool = 512 * 6 * 5

def main():
    # load the data
    from CelebADataset import CelebADataset
    mytransform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                    ])
    face_data = CelebADataset(csv_file='celeba_label/identity_CelebA.txt', 
                              root_dir='img_align_celeba',
                              transform=mytransform)
    train_loader = Data.DataLoader(dataset=face_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # load the model
    from my_vgg19_b import 
    model = my_vgg19_b(num_classes=num_people, pic_size=pic_after_MaxPool)
    model.load_state_dict(torch.load('c_params.pkl'))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # start here

if __name__ == '__main__':
    main()