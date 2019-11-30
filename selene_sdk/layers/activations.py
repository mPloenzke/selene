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


class ReEF(nn.Module):
    """
    Rectified Exponential Function

    Parameters
    ----------
    self : tensor
        input tensor

    Returns
    -------
    Activated tensor using rectified exponential 

    """
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input.clamp(0).exp()

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
        return input.pow(self.power).clamp(0)

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

class shifted_exponential_activation(nn.Module):
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
    def __init__(self, shift = 0, scale = 1):
        super().__init__()
        self.shift = shift
        self.scale = scale
    def forward(self, input):
        #return input.mul(self.scale).add(self.shift).exp().add(-1)
        #return input.mul(self.scale).add(self.shift).exp()
        return input.exp().mul(self.scale).add(self.shift)

class shifted_relu_activation(nn.Module):
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
    def __init__(self, shift = 0, scale = 1):
        super().__init__()
        self.shift = shift
        self.scale = scale
    def forward(self, input):
        return input.clamp(0).mul(self.scale).add(self.shift)

class shifted_sigmoid_activation(nn.Module):
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
    def __init__(self, shift = 0, scale = 1):
        super().__init__()
        self.shift = shift
        self.scale = scale
    def forward(self, input):
        return input.add(self.shift).sigmoid().mul(self.scale)

class shifted_tanh_activation(nn.Module):
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
    def __init__(self, shift = 0, scale = 1):
        super().__init__()
        self.shift = shift
        self.scale = scale
    def forward(self, input):
        return input.add(self.shift).tanh().mul(250).add(self.scale)
