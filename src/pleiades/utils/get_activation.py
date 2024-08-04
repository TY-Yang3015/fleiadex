import flax.linen as nn
from functools import partial


def get_activation(activation):
    if activation == "relu":
        return nn.relu
    elif activation == "sigmoid":
        return nn.sigmoid
    elif activation == "tanh":
        return nn.tanh
    elif activation == "silu":
        return nn.silu
    elif activation == "gelu":
        return nn.gelu
    elif activation == "softmax":
        return nn.softmax
    elif activation == "softplus":
        return nn.softplus
    elif activation == 'leaky_relu':
        return partial(nn.leaky_relu, negative_slope=0.1)
    elif activation == 'elu':
        return nn.elu
    else:
        raise NotImplementedError("only 'relu', 'sigmoid'"
                                  ", 'tanh', 'silu', "
                                  "'softmax', 'softplus', 'leaky_relu'"
                                  "'gelu' and 'elu' are"
                                  " supported.")
