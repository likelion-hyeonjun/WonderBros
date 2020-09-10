import torch.nn as nn
import torch.nn.functional as F
import torch


class stn_conv(nn.Module):
    def __init__(self, height, width, channel): #inc, outc 이거 알아서 파라미터로 받아보기
        super(stn_conv, self).__init__()
        self.height = height
        self.width = width
        self.localization = nn.Sequential(
            nn.Conv2d(channel, 10, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 15, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        ) 

        self.fc_loc = nn.Sequential(
            nn.Linear(40560, 128),
            nn.Tanh(),
            nn.Linear(128, 3*2), #2-layer MLP
            nn.Tanh()
            )
        #localization -> fc_loc 으로 파라미터 맞추는 네트워크 .
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 40560) #이거 크기 fc_loc 첫번재랑 맞춰주기)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        return self.stn(x)