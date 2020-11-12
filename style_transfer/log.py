import torch
from torch.utils.tensorboard import SummaryWriter
from time import time

class Logger(SummaryWriter):
    def __init__(self, iter):
        super(Logger, self).__init__()
        self.iter = iter
        self.time = time()

    def __call__(self, iter, losses, artwork, scaler):
        iter_time = (time() - self.time) / (iter - self.iter)
        total_loss, content_loss, style_loss = [loss.item()
                                                for loss in losses[0:3]]
        content_losses, style_losses = losses[3:]
        mean_abs_grad = torch.abs(artwork.grad).mean().item()
        zeros_grad = (artwork.grad == 0).sum().item()
        print(f'Iteration: {iter:<10}'
              f'Time: {iter_time:<15.3}'
              f'Loss: {total_loss:<15.1e}'
              f'Content Loss: {content_loss:<15.1e}'
              f'Style Loss: {style_loss:<15.1e}'
              f'Mean Abs Grad: {mean_abs_grad:<15.1e}'
              f'Grad Zeros: {zeros_grad:<5}'
              f'Grad Scale: {scaler.scale_factor:<5.1e}')
        self.add_scalar('Losses/Total Loss', total_loss, iter)
        self.add_scalar('Losses/Content Loss', content_loss, iter)
        self.add_scalar('Losses/Style Loss', style_loss, iter)
        for layer, loss in content_losses.items():
            self.add_scalar(f'Content Losses/{layer}', loss.item(), iter)
        for layer, loss in style_losses.items():
            self.add_scalar(f'Style Losses/{layer}', loss.item(), iter)
        self.add_scalar('Gradients/Mean Abs Grad', mean_abs_grad, iter)
        self.iter, self.time = iter, time()
