import torch
from .log import Logger
from .data import Preprocess, Postprocess
from .loss import VGG19Loss
from .amp import GradScaler

class StyleTransfer(object):
    def __init__(self, content_weight=1, style_weight=1e3,
                 content_weights="{'relu_4_2':1}", style_weights=("{'relu_1_1':"
                 "1,'relu_2_1':1,'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"),
                 avg_pool=False, feature_norm=True, weights='original',
                 preserve_color='style', device='auto', use_amp=False, 
                 logging=50):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preprocess = Preprocess(preserve_color, device)
        self.postprocess = Postprocess(device)
        self.criterion = VGG19Loss(content_weight, style_weight,
                                   content_weights, style_weights, avg_pool,
                                   feature_norm, weights, device)
        self.use_amp = device == 'cuda' and use_amp
        self.logging = logging

    def __call__(self, content, style, area=512, init_random=False,
                 init_img=None, iter=500, lr=1, adam=False):
        assert not (init_random and init_img)
        artwork, optimizer, scaler = self._init_call(content, style, area, 
                                                     init_random, init_img, 
                                                     lr, adam)
        i = 0
        if self.logging:
            logger = Logger(i)
        while i <= iter:
            def closure():
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    losses = self.criterion(artwork)
                    total_loss = losses[0] if adam else lr * losses[0]
                scaler.scale(total_loss).backward()
                scaler.unscale(optimizer)
                nonlocal i
                i += 1
                if self.logging and (i % self.logging == 0):
                    logger(i, losses, artwork, scaler)
                return total_loss
            optimizer.step(closure)
        if self.logging:
            logger.close()
        self.criterion.reset()
        return self.postprocess(artwork)

    def _init_call(self, content, style, area, init_random, init_img, lr, adam):
        content_tensor, style_tensor = self.preprocess(content, area, style)
        self.criterion.set_targets(content_tensor, style_tensor)
        if init_random:
            artwork = torch.randn_like(content_tensor)
        elif init_img:
            artwork = self.preprocess(init_img, list(content_tensor.size()[2:]))
        else:
            artwork = content_tensor.clone()
        artwork.requires_grad_()
        if adam:
            optimizer = torch.optim.Adam([artwork], lr=lr)    
        else:
            optimizer = torch.optim.LBFGS([artwork])
        scaler = GradScaler(2.**16, 2.0, 0.5, 50, self.use_amp)
        return artwork, optimizer, scaler
