import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from ast import literal_eval
from itertools import chain
from .utils import gram_matrix

class ContentLoss(nn.Module):
    def __init__(self, mode):
        super(ContentLoss, self).__init__()
        self.mode = mode

    def forward(self, input):
        if self.mode == 'loss' and input.size() == self.target.size():
            self.loss = F.mse_loss(input, self.target)
        elif self.mode == 'target':
            self.target = input
        return input

class StyleLoss(nn.Module):
    def __init__(self, mode, feature_norm):
        super(StyleLoss, self).__init__()
        self.mode = mode
        self.feature_norm = feature_norm

    def forward(self, input):
        if self.mode == 'loss':
            self.loss = F.mse_loss(gram_matrix(input, self.feature_norm),
                                   self.target)
        elif self.mode == 'target':
            self.target = gram_matrix(input, self.feature_norm)
        return input

class VGG19Loss(nn.Module):
    def __init__(self, content_weight, style_weight, content_weights,
                 style_weights, avg_pool, feature_norm, weights, device):
        super(VGG19Loss, self).__init__()
        content_weights = literal_eval(content_weights)
        style_weights = literal_eval(style_weights)
        self.content_weight, self.style_weight = content_weight, style_weight
        self.style_weights = {layer: weight / sum(style_weights.values())
                              for layer, weight in style_weights.items()}
        self.content_weights = {layer: weight / sum(content_weights.values())
                                for layer, weight in content_weights.items()}
        self._build_vgg_loss(avg_pool, feature_norm, weights, device)

    def forward(self, input):
        self.vgg_loss(input)
        content_loss, style_loss = 0, 0
        content_losses, style_losses = {}, {}
        for layer in self.content_weights:
            content_losses[layer] = self.content_losses[layer].loss
            content_loss += content_losses[layer] * self.content_weights[layer]
        for layer in self.style_weights:
            style_losses[layer] = self.style_losses[layer].loss
            style_loss += style_losses[layer] * self.style_weights[layer]
        total_loss = content_loss * self.content_weight + \
                     style_loss * self.style_weight
        return (total_loss, content_loss, style_loss,
                content_losses, style_losses)

    def set_targets(self, content, style):
        self._set_modes('target', 'none')
        self.vgg_loss(content)
        self._set_modes('none', 'target')
        self.vgg_loss(style)
        self._set_modes('loss', 'loss')

    def reset(self):
        for loss in chain(self.content_losses.values(),
                          self.style_losses.values()):
            if hasattr(loss, 'target'): delattr(loss, 'target')
            if hasattr(loss, 'loss'): delattr(loss, 'loss')
        self._set_modes('none', 'none')

    def _set_modes(self, content_mode, style_mode):
        for loss in self.content_losses.values():
            loss.mode = content_mode
        for loss in self.style_losses.values():
            loss.mode = style_mode

    def _build_vgg_loss(self, avg_pool, feature_norm, weights, device):
        self.content_losses, self.style_losses = {}, {}
        self.vgg_loss = nn.Sequential()
        vgg = models.vgg19(pretrained=False).features
        if weights in ('original', 'normalized'):
            state_dict = load_state_dict_from_url('https://storage.googleapis'
                         f'.com/prism-weights/vgg19-{weights}.pth')
        else:
            state_dict = torch.load(weights)
        vgg.load_state_dict(state_dict)
        vgg = vgg.eval()
        for param in vgg.parameters():
            param.requires_grad_(False)
        i_pool, i_conv = 1, 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i_conv += 1
                name = f'conv_{i_pool}_{i_conv}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i_pool}_{i_conv}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i_pool}'
                if avg_pool:
                    layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
                i_pool += 1
                i_conv = 0
            self.vgg_loss.add_module(name, layer)
            if name in self.content_weights:
                content_loss = ContentLoss('none')
                self.vgg_loss.add_module(f'content_loss_{i_pool}_{i_conv}',
                                         content_loss)
                self.content_losses[name] = content_loss
            if name in self.style_weights:
                style_loss = StyleLoss('none', feature_norm)
                self.vgg_loss.add_module(f'style_loss_{i_pool}_{i_conv}',
                                         style_loss)
                self.style_losses[name] = style_loss
            if (len(self.style_weights) == len(self.style_losses) and
               len(self.content_weights) == len(self.content_losses)):
               break
        self.vgg_loss.to(device)
