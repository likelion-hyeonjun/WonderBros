import torch
import torch.nn as nn
from models import models

class EarlyStopping():
    def __init__(self, saved_model, patience = 0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
        self.model_name = saved_model
    
    def validate(self,loss,model):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("Training stopped early")
                return True
        else:
            torch.save(model.state_dict(), '{}_best.pkl'.format(self.model_name)) #이거 early stopping할때 멈추게 하기 
            self._step = 0
            self._loss = loss
        return False


def freezeResNet(model):
    for name, p in model.named_parameters():
        if 'fc' not in name:
            p.required_grad = False
    return model #이거 필요한가?

def freezeDenseNet(model):
    for name, p in model.named_parameters():
        if 'classifier' not in name:
            p.required_grad = False
    return model

def freezeVGG(model):
    for name, p in model.named_parameters():
        if 'classifier.6' not in name:
            p.required_grad = False
    return model


def fineTuningModel(name, num_classes, is_freeze, pretrained = True): #freeze #true true시 r

    model = models(name, pretrained) 
    if 'resnet' in name:
        input_features = model.fc.in_features
        model.fc = nn.Linear(in_features = input_features, out_features = num_classes, bias=True)
        if is_freeze and pretrained :
            model = freezeResNet(model)
    elif 'vgg' in name:
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features = input_features, out_features = num_classes, bias=True)
        if is_freeze and pretrained:
            model = freezeVGG(model)
    elif 'densenet' in name:
        input_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features = input_features, out_features = num_classes, bias=True)
        if is_freeze and pretrained:
            model = freezeDenseNet(model)
    
    return model

def init_conv_offset(m):
    m.weight.data = torch.zeros_like(m.weight.data)
    if m.bias is not None:
        m.bias.data = torch.FloatTensor(m.bias.shape[0]).zero_()

