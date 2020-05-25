import torch

from ._functions import Scatter


def scatter(inputs, target_gpus, dim=0):
    """Scatter inputs to target gpus.
    """

    def scatter_map(obj, self_def_scatter_flag=0):
        """
            self_def_scatter_flag: 0
        """

        if self_def_scatter_flag == 1 and (isinstance(obj, torch.Tensor) or isinstance(obj, list)):
            return Scatter.forward(target_gpus, obj)
        if self_def_scatter_flag == 2 and isinstance(obj, list):
            return obj

        if isinstance(obj, tuple) and len(obj) > 0:
            self_def_scatter_flag = [self_def_scatter_flag] * len(obj)
            return list(zip(*map(scatter_map, obj, self_def_scatter_flag)))
        if isinstance(obj, list) and len(obj) > 0:
            self_def_scatter_flag = [self_def_scatter_flag] * len(obj)
            out = list(map(list, zip(*map(scatter_map, obj, self_def_scatter_flag))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            flag = []
            for key in obj.keys():
                if key in ('img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'):
                    flag.append(1)
                elif key in ('img_meta', 'gt_masks'):
                    flag.append(2)
                else:
                    flag.append(0)

            out = list(map(type(obj), zip(*map(scatter_map, obj.items(), flag))))
            return out
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
