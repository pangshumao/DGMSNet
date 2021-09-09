from functools import wraps
import numpy as np
from tqdm import tqdm as _tqdm
import matplotlib.pyplot as plt
import torch
import os

@wraps(_tqdm)
def tqdm(*args, **kwargs):
    with _tqdm(*args, **kwargs) as t:
        try:
            for _ in t:
                yield _
        except KeyboardInterrupt:
            t.close()
            raise KeyboardInterrupt

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def exp_decay(current, gamma):
    return np.exp(-gamma * current)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    # if not os.path.exists(checkpoint_path):
    #     raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    #
    # state = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(state['model_state_dict'])
    #
    # if optimizer is not None:
    #     optimizer.load_state_dict(state['optimizer_state_dict'])
    #
    # return state

    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return pretrained_dict


if __name__ == '__main__':
    epochs = 200.0
    ts = np.arange(0, epochs, 1)
    out = []
    for t in ts:
        # out.append(get_current_consistency_weight(t, 0.001, 20))
        # out.append(cosine_rampdown(t, 200))
        out.append(exp_decay(t, gamma=1./(epochs / 2.0)))
    out = np.array(out)
    plt.plot(ts, out)
    plt.show()
