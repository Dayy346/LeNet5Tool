import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)  # Number of input connections
        bound = 2.4 / fan_in
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
