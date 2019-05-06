"""
Useful activation functions.

"""
import torch
import torch.nn as nn

class Exponential_activation(nn.Module):
    """
    Exponential activation function

    Parameters
    ----------
    self : tensor
        input tensor

    Returns
    -------
    Activated tensor using exponential function

    """
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input.exp()

class Polynomial_activation(nn.Module):
    """
    Polynomial activation function

    Parameters
    ----------
    self : tensor
        input tensor
    power : int
        Power of polynomial

    Returns
    -------
    Activated tensor using polynomial of degree power

    """
    def __init__(self, power = 3):
        super().__init__()
        self.power = power
    def forward(self, input):
        return input.pow(self.power)

class Bernoulli_Dropout(nn.Module):
    """
    Activation-based dropout

    Parameters
    ----------
    self : tensor
        input tensor

    Returns
    -------
    Clamped tensor using activation values as probabilities of dropout

    """
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input*(input.sigmoid().bernoulli())

class Linear_activation(nn.Module):
    """
    Linear activation function. 

    Parameters
    ----------
    self : tensor
        input tensor

    Returns
    -------
    Tensor

    """
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input
