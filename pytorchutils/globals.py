"""Global imports and constants"""

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ['torch', 'nn', 'DEVICE']
