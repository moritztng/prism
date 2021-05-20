class GradScaler(object):
    def __init__(self, init_scale, growth_factor, backoff_factor,  
                 growth_interval, enabled):
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.enabled = enabled
        self.unskipped_iter = 0

    def scale(self, outputs):
        if not self.enabled:
            return outputs
        return self.scale_factor * outputs
    
    def unscale(self, optimizer):
        if not self.enabled:
            return 
        for param in optimizer.param_groups[0]['params']:
            if param.grad.isnan().any() or param.grad.isinf().any():
                optimizer.zero_grad()
                self.scale_factor *= self.backoff_factor
                self.unskipped_iter = 0
                return
        for param in optimizer.param_groups[0]['params']:
            param.grad /= self.scale_factor
        self.unskipped_iter += 1
        if self.unskipped_iter >= self.growth_interval:
            self.scale_factor *= self.growth_factor
            self.unskipped_iter = 0
