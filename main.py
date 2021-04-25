import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from deformable_filter import *
from stn_conv import *
from data.CUB_dataset import CubDataset
from model.resnet_STN import resnet_multi_stn101

import os
import argparse

from models import *
from utils import fineTuningModel
from train import train
from test import Test

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='GroceryDataset Training with VGG, ResNet and DenseNet')
parser.add_argument('--model', 
                    default ='vgg19',
                    help='choose model for experiment \n available params: densenet, resnet, vgg19')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--train_batch_size', type=int, default=1, help='# images in batch when training')
parser.add_argument('--test_batch_size', type=int, default=1, help='# images in batch when testing')
parser.add_argument('--opt', default='SGD', help='optimizer')
parser.add_argument('--freeze', type=str2bool ,default="true", help='true: fine-tuning for only classifier layer, false: fine-tuning for whole layer (with pretrained)')

args = parser.parse_args()

print('==> Preparing data..')

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(280),
    transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomVerticalFlip(0.5),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     transforms.Normalize([0.485, 0.465, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = CubDataset(transform=train_transform)
test_dataset = CubDataset(transform=test_transform, test=True)
num_test_data = (int)(len(test_dataset) * 0.8)
num_valid_data = len(test_dataset) - num_test_data # cause #test almost equal with trainset
test_dataset , valid_dataset = torch.utils.data.random_split(test_dataset, (num_test_data, num_valid_data))

device = 'cuda' if torch.cuda.is_available() else 'cpu'



# preprocess_resnet = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])



# trainDataPath = trainDataPath = os.getcwd()+"/GroceryStoreDataset-master/dataset/train/Packages"
# train_dataset = ImageFolder(root=trainDataPath, transform = preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.train_batch_size,
                                            shuffle = True,
                                            num_workers=4)
# testDataPath = os.getcwd()+"/GroceryStoreDataset-master/dataset/test/Packages"
# test_dataset = ImageFolder(root=testDataPath, transform = preprocess) #이거 test시에해야하나?
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = args.test_batch_size,
                                          shuffle = False,
                                          num_workers=4)
# validDataPath = os.getcwd()+"/GroceryStoreDataset-master/dataset/val/Packages"
# valid_dataset = ImageFolder(root=validDataPath, transform = preprocess)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size = args.test_batch_size,
                                           shuffle = False,
                                           num_workers=4)

# Grocery dataset을 사용할때 필요한 classes들. 
# classes = ('Alpro-Blueberry-Soyghurt','Alpro-Fresh-Soy-Milk',
#             'Alpro-Shelf-Soy-Milk', 'Alpro-Vanilla-Soyghurt',
#             'Arla-Ecological-Medium-Fat-Milk', 'Arla-Ecological-Sour-Cream',
#             'Arla-Lactose-Medium-Fat-Milk', 'Arla-Medium-Fat-Milk',
#             'Arla-Mild-Vanilla-Yoghurt', 'Arla-Natural-Mild-Low-Fat-Yoghurt',
#             'Arla-Natural-Yoghurt','Arla-Sour-Cream',
#             'Arla-Sour-Milk','Arla-Standard-Milk',
#             'Bravo-Apple-Juice', 'Bravo-Orange-Juice',
#             'Garant-Ecological-Medium-Fat-Milk', 'Garant-Ecological-Standard-Milk',
#             'God-Morgon-Apple-Juice', 'God-Morgon-Orange-Juice',
#             'God-Morgon-Orange-Red-Grapefruit-Juice', 'God-Morgon-Red-Grapefruit-Juice',
#             'Oatly-Natural-Oatghurt', 'Oatly-Oat-Milk',
#             'Tropicana-Apple-Juice', 'Tropicana-Golden-Grapefruit',
#             'Tropicana-Juice-Smooth', 'Tropicana-Mandarin-Morning',
#             'Valio-Vanilla-Yoghurt', 'Yoggi-Strawberry-Yoghurt',
#             'Yoggi-Vanilla-Yoghurt')



model = fineTuningModel(args.model, 200, args.freeze, True) 
#STN model 사용시 아래와 같은 코드를 통해 model 넣어주기 .
#model = resnet_multi_stn101(pretrained=True, num_classes=200, p=0)

# model slightly chaning part
# deformable 로 바꾸기 위해선 model.layer4[2].conv = deformable_filter(512,512) 와 같이 input/ out channel 맞춰서 넣어주기 
if args.model == 'resnet101':
    model.layer4[2].conv2 = deformable_filter(512,512)
    


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

saved_model = 'checkpoints/for_test'

train(model = model, 
    train_loader = train_loader, 
    valid_loader = valid_loader,
    criterion = criterion, 
    optimizer = optimizer, 
    scheduler = exp_lr_scheduler, 
    device = device, 
    num_train_data =len(train_dataset), 
    num_valid_data = len(valid_dataset), 
    saved_model= saved_model,
    num_epochs = args.epoch)
model.load_state_dict(torch.load('{}_best.pkl'.format(saved_model)))
test_model = Test(model, test_loader, len(test_dataset))
test_model.OverallAccuracy()
#test_model.ClassAccuracy(classes)