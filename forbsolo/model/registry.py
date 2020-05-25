from forbsolo.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
GRIDS = Registry('grid')
CRITERIA = Registry('criterion')
LOSSES = Registry('loss')
MODELS = Registry('model')
