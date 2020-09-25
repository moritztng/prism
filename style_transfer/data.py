import torch
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize,
                                    Lambda, ToPILImage)
from .utils import match_color

mean = torch.Tensor([0.40760392, 0.45795686, 0.48501961])
std = torch.Tensor([1, 1, 1])

class Preprocess(object):
    def __init__(self, preserve_color, device):
        self.to_tensor = Compose([Resize(0),
                                  ToTensor()])
        self.normalize = Compose([Lambda(lambda x: x.flip(0)),
                                  Normalize(mean, std),
                                  Lambda(lambda x: x * 255),
                                  Lambda(lambda x: x.unsqueeze(0)),
                                  Lambda(lambda x: x.to(device))])
        self.preserve_color = preserve_color

    def __call__(self, content, size, style=None):
        self.to_tensor.transforms[0].size = size
        content_tensor = self.to_tensor(content)
        if style:
            style_tensor = self.to_tensor(style)
            if self.preserve_color == 'style':
                content_tensor = match_color(content_tensor, style_tensor)
            elif self.preserve_color == 'content':
                style_tensor = match_color(style_tensor, content_tensor)
            style_tensor = self.normalize(style_tensor)
        content_tensor = self.normalize(content_tensor)
        return (content_tensor, style_tensor) if style else content_tensor

class Postprocess(object):
    def __init__(self):
        self.transform = Compose([Lambda(lambda x: x / 255),
                                  Normalize(-mean, std),
                                  Lambda(lambda x: x.clamp(0,1)),
                                  Lambda(lambda x: x.flip(0)),
                                  ToPILImage()])

    def __call__(self, img):
        return self.transform(img.squeeze().detach().cpu())
