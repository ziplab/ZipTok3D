import torch


class AxisScaling:
    def __init__(self, interval=(0.75, 1.25), jitter=True, eps=1e-10):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        self.eps = eps

    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        if point is not None:
            point = point * scaling

        ## TODO: clamping
        max_val = max(torch.abs(surface).max().item(), 0.1)
        scale = (1 / max_val) * 0.999999
        surface *= scale
        if point is not None:
            point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point, max_val
