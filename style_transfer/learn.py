import torch
from .log import Logger
from .data import Preprocess, Postprocess
from .loss import VGG19Loss
from .amp import GradScaler

class StyleTransfer(object):
    """The central object that performs style transfer when called.
    
    :param lr: Learning rate of the optimizer.
    :type lr: float
    :param content_weight: Weight for content loss.
    :type content_weight: int
    :param style_weight: Weight for style loss.
    :type style_weight: int
    :param content_weights: Weights of content loss for each layer.
    :type content_weights: str
    :param style_weights: Weights of style loss for each layer.
    :type style_weights: str
    :param avg_pool: If ``True``, replaces max-pooling by average-pooling.
    :type avg_pool: bool
    :param feature_norm: If ``True``, divides each ``style_weight`` by the 
        square of the number of feature maps in the corresponding layer.
    :type feature_norm: bool
    :param weights: Weights of the VGG19 Network. Either ``'original'`` 
        weights or ``'normalized'`` weights that are scaled such that the 
        mean activation of each convolutional filter over images
        and positions is equal to one.
    :type weights: str
    :param preserve_color: If set to ``'style'``, changes color of content 
        image to color of style image. If set to ``'content'``, changes 
        color of style image to color of content image. If set to 
        ``None``, does not change any color.
    :type preserve_color: Union[str, None]
    :param device: Set to ``'cpu'`` to use CPU, ``'cuda'`` to use GPU  or 
        ``'auto'`` to automatically choose device.
    :type device: str
    :param use_amp: If ``True``, uses automatic mixed precision for 
        training.
    :type use_amp: bool
    :param adam: If ``True``, uses Adam instead of LBFGS optimizer.
    :type adam: bool
    :param optim_cpu: If ``True``, optimizes artwork on CPU, but can 
        calculate gradients on GPU. This moves some data from GPU memory 
        to working memory, but increases training time.
    :type optim_cpu: bool
    :param logging: Number of iterations between logs. If set to ``0``, does 
        not log.
    :type logging: int
    """

    def __init__(self, lr=1, content_weight=1, style_weight=1e3,
                 content_weights="{'relu_4_2':1}", style_weights=("{'relu_1_1':"
                 "1,'relu_2_1':1,'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"),
                 avg_pool=False, feature_norm=True, weights='original',
                 preserve_color='style', device='auto', use_amp=False, 
                 adam=False, optim_cpu=False, logging=50):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.preprocess = Preprocess(preserve_color, self.device)
        self.postprocess = Postprocess()
        self.criterion = VGG19Loss(content_weight, style_weight,
                                   content_weights, style_weights, avg_pool,
                                   feature_norm, weights, self.device)
        self.lr = lr
        self.use_amp = self.device == 'cuda' and use_amp
        self.adam = adam
        self.optim_cpu = optim_cpu
        self.logging = logging

    def __call__(self, content, style, area=512, init_random=False,
                 init_img=None, iter=500):
        """Returns artwork created from content and style image.

        :param content: Content image.
        :type content: PIL.Image.Image
        :param style: Style image.
        :type style: PIL.Image.Image
        :param area: ``content`` and ``style`` are scaled such that their area 
            is ``area`` * ``area``. ``artwork`` has the same shape as 
            ``content``.
        :type area: int
        :param init_random: If ``True``, initializes ``artwork`` with random 
            image.
        :type init_random: bool
        :param init_img: Image with which ``artwork`` is initialized. If set 
            to ``None``, initializes ``artwork`` either with ``content`` or 
            randomly.
        :type init_img: Union[PIL.Image.Image, None]
        :param iter: Number of iterations.
        :type iter: int
        :return: ``artwork`` that matches content of ``content`` and style of 
            ``style``.
        :rtype: PIL.Image.Image
        """

        assert not (init_random and init_img)
        artwork, optimizer, scaler = self._init_call(content, style, area, 
                                                     init_random, init_img)
        i = 0
        if self.logging:
            logger = Logger(i)
        while i <= iter:
            def closure():
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    losses = self.criterion(artwork.to(self.device))
                    total_loss = losses[0] if self.adam else self.lr*losses[0]
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

    def _init_call(self, content, style, area, init_random, init_img):
        content_tensor, style_tensor = self.preprocess(content, area, style)
        self.criterion.set_targets(content_tensor, style_tensor)
        if init_random:
            artwork = torch.randn_like(content_tensor)
        elif init_img:
            artwork = self.preprocess(init_img, list(content_tensor.size()[2:]))
        else:
            artwork = content_tensor.clone()
        if self.optim_cpu:
            artwork = artwork.cpu()
        artwork.requires_grad_()
        if self.adam:
            optimizer = torch.optim.Adam([artwork], lr=self.lr)    
        else:
            optimizer = torch.optim.LBFGS([artwork])
        scaler = GradScaler(2.**16, 2.0, 0.5, 50, self.use_amp)
        return artwork, optimizer, scaler
