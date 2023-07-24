import torch
from torch import nn 
from .register import MODELS
from .models import ResNet, InternImage, ConvNeXt, ViT


def get_model(hparams):
    return Classfier(hparams.model, hparams.model.out_func)

class Classfier(nn.Module):
    def __init__(self, model, func = 'sigmoid') -> None:
        super(Classfier, self).__init__()
        self.model = MODELS.get(model.name)(**model)
        self.func = nn.Sequential()
        if func == 'sigmoid':
            self.func.append(nn.Sigmoid())
        if func == 'softmax':
            self.func.append(nn.Softmax(dim=-1))

    def forward(self, input):

        pre_cls = self.model(input)
        output = self.func(pre_cls)

        return output