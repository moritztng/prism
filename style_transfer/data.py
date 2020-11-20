import torch
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize,
                                    Lambda, ToPILImage)
from .utils import match_color

mean = torch.Tensor([0.40760392, 0.45795686, 0.48501961])
std = torch.Tensor([1, 1, 1])

class Preprocess(object):
    def __init__(self, preserve_color, device):
        self.normalize = Compose([Lambda(lambda x: x.flip(0)),
                                  Normalize(mean, std),
                                  Lambda(lambda x: x * 255),
                                  Lambda(lambda x: x.unsqueeze(0))])
        self.device = device
        self.preserve_color = preserve_color

    def __call__(self, content, size, style=None):
        content_tensor = self._to_tensor(content, size)
        if style:
            style_tensor = self._to_tensor(style, size)
            if self.preserve_color == 'style':
                content_tensor = match_color(content_tensor, style_tensor)
            elif self.preserve_color == 'content':
                style_tensor = match_color(style_tensor, content_tensor)
            style_tensor = self.normalize(style_tensor)
        content_tensor = self.normalize(content_tensor)
        return (content_tensor, style_tensor) if style else content_tensor

    def _to_tensor(self, img, size):
        if isinstance(size, int):
            resize_factor = (size**2 / (img.size[0] * img.size[1]))**(1/2)
            size = int(min(img.size) * resize_factor)
        return Compose([Resize(size), ToTensor()])(img).to(self.device)

class Postprocess(object):
    def __init__(self):
        self.transform = Compose([Lambda(lambda x: x / 255),
                                  Normalize(-mean, std),
                                  Lambda(lambda x: x.clamp(0,1)),
                                  Lambda(lambda x: x.flip(0)),
                                  Lambda(lambda x: x.cpu()),
                                  ToPILImage()])

    def __call__(self, img):
        return self.transform(img.squeeze().detach())
