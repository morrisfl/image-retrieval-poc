import torch
from torch.nn import Module


class DINOModel(Module):
    def __init__(self, model_name):
        super(DINOModel, self).__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)

    def forward(self, x):
        x = self.model(x)
        return x


class DINOv2Model(Module):
    def __init__(self, model_name):
        super(DINOv2Model, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward(self, x):
        x = self.model(x)
        return x
