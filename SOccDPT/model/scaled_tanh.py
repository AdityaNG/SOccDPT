import torch.nn as nn
import torch.nn.functional as F

class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()

    def forward(self, x):
        x = F.tanh(x)  # Apply tanh activation
        return 0.5 * x + 0.5  # Scale and shift
