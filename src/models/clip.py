import open_clip
from torch.nn import Module


class CLIPModelPretrained(Module):
    def __init__(self, model_name, pretrained_model):
        super(CLIPModelPretrained, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained_model)

    def forward(self, x):
        x = self.model.encode_image(x)
        return x
