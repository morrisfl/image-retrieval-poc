import clip
import open_clip
import torch
from torch.nn import Module


class CLIPModel(Module):
    def __init__(self, model_name, device, jit=False):
        super(CLIPModel, self).__init__()
        self.model, self.preprocess = clip.load(model_name, device=device, jit=jit)

    def forward(self, x: torch.Tensor):
        x = self.model.encode_image(x)
        return x


class CLIPModelPretrained(Module):
    def __init__(self, model_name, pretrained_model):
        super(CLIPModelPretrained, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained_model)

    def forward(self, x):
        x = self.model.encode_image(x)
        return x
