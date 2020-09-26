import torch
from .log import Logger
from .data import Preprocess, Postprocess
from .loss import VGG19Loss

class StyleTransfer(object):
    def __init__(self, lr=1, content_weight=1, style_weight=1e3,
                 content_weights="{'relu_4_2':1}", style_weights=("{'relu_1_1':"
                 "1,'relu_2_1':1,'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"),
                 avg_pool=False, feature_norm=True, weights='original',
                 preserve_color='style', device='auto', logging=50):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preprocess = Preprocess(preserve_color, device)
        self.postprocess = Postprocess(device)
        self.criterion = VGG19Loss(content_weight, style_weight,
                                   content_weights, style_weights, avg_pool,
                                   feature_norm, weights, device)
        self.lr = lr
        self.logging = logging

    def __call__(self, content, style, size=400, init_random=False,
                 init_img=None, iter=500):
        assert not (init_random and init_img)
        artwork, optimizer = self._init_artwork_criterion_optimizer(content,
                             style, size, init_random, init_img)
        i = 0
        if self.logging:
            logger = Logger(i)
        while i <= iter:
            def closure():
                optimizer.zero_grad()
                losses = self.criterion(artwork)
                total_loss = losses[0]
                scaled_loss = self.lr * total_loss
                scaled_loss.backward()
                nonlocal i
                i += 1
                if self.logging and (i % self.logging == 0):
                    logger(i, losses, artwork)
                return total_loss
            optimizer.step(closure)
        if self.logging:
            logger.close()
        self.criterion.reset()
        return self.postprocess(artwork)

    def _init_artwork_criterion_optimizer(self, content, style, size,
                                          init_random, init_img):
        content_tensor, style_tensor = self.preprocess(content, size, style)
        self.criterion.set_targets(content_tensor, style_tensor)
        if init_random:
            artwork = torch.randn_like(content_tensor)
        elif init_img:
            artwork = self.preprocess(init_img, list(content_tensor.size()[2:]))
        else:
            artwork = content_tensor.clone()
        artwork.requires_grad_()
        return artwork, torch.optim.LBFGS([artwork])
