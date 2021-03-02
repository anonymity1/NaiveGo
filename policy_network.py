import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Net(nn.Module):
    def __init__(self, board_width, board_height):
        pass

    def forward(self, board):
        pass

class PolicyNetwork():
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False):
        pass

    def policy(self):
        pass

    def train_step(self):
        pass

    def get_policy_param(self):
        pass

    def save_model(self):
        pass

if __name__ == '__main__':
    pass