import torch.nn as nn
from deform_conv import DeformConv2D
from utils import init_conv_offset

class deformable_filter(nn.Module):
    def __init__(self, in_c, out_c): #inc, outc 이거 알아서 파라미터로 받아보기
        super(deformable_filter, self).__init__()
        self.offset = nn.Conv2d(in_c, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2D(in_c, out_c, kernel_size=3, padding=1)
        init_conv_offset(self.offset)
        #init_weights(self.deform_conv)
    
    def forward(self, x):
        offset = self.offset(x)
        x = self.deform_conv(x, offset)
        
        return x