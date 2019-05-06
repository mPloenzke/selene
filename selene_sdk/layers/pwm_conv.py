"""
Utilities for loading configurations, instantiating Python objects, and
running operations in _Selene_.

"""
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple

from Bio import motifs

def pwm_conv1d_weight(input, ppm, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Gradient calculation for the filter weights of the PWM-based convolution.

    Parameters
    ----------
    input : tensor
        input tensor
    ppm : tensor
        Position probability matrix
    grad_output : tensor
        Gradient output from backwards pass

    Returns
    -------
    Calculated gradients

    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]
    weight_size = ppm.shape

    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1)
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2])

    grad_weight = torch.conv1d(input, grad_output, None, dilation, padding,
                               stride, in_channels * min_batch)

    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2])

    grad_weight_sum = grad_weight.sum(dim=0).view(in_channels // groups, out_channels, grad_weight.shape[2]).transpose(
            0, 1).narrow(2, 0, weight_size[2])

    weights = ppm - (ppm**2)
    return grad_weight_sum.div(np.log(2)*weights)

def pwm_conv1d_bias(input, weights, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Gradient calculation for the filter offsets of the PWM-based convolution.

    Parameters
    ----------
    input : tensor
        input tensor
    weights : tensor
        weights matrix
    grad_output : tensor
        Gradient output from backwards pass

    Returns
    -------
    Calculated gradients

    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]
    weight_size = weights.shape

    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1)
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2])*0+1

    grad_weight = torch.conv1d(input, grad_output, None, dilation, padding,
                               stride, in_channels * min_batch)

    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2])

    grad_weight_sum = grad_weight.sum(dim=0).view(in_channels // groups, out_channels, grad_weight.shape[2]).transpose(0, 1).narrow(2, 0, weight_size[2])
    return grad_weight_sum.mean(dim=[1,2])

def pwm_conv1d_input(input, weight, bg_tensor, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Gradient calculation for the filter weights of the PWM-based convolution.

    Parameters
    ----------
    input : tensor
        input tensor
    weight : tensor
        weight tensor
    bg_tensor : tensor
        background probabilities tensor
    grad_output : tensor
        Gradient output from backwards pass

    Returns
    -------
    Calculated gradients

    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    kernel_size = [weight.shape[2]]
    input_size = input.shape

    if input_size is None:
        raise ValueError("grad.conv1d_input requires specifying an input_size")

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size)

    pwm = weight.div(bg_tensor).log2()
    return torch.conv_transpose1d(
        grad_output, pwm, None, stride, padding, grad_input_padding, groups,
        dilation)

class pwmConv(torch.autograd.Function):
  """
    Torch autograd function specifying backward and forward passes.
  """
  @staticmethod
  def forward(ctx, x, w, bg, bias, stride=1, padding=0, dilation=1, groups=1):
    ppm = w.exp()
    ppm = ppm.div(ppm.sum(dim=1,keepdim=True))
    ctx.save_for_backward(x, ppm, bg)
    pwm = ppm.double().div(bg).log2().float()
    return F.conv1d(x, pwm, bias, stride, padding, dilation, groups)

  @staticmethod
  def backward(ctx, grad_output, stride=1, padding=0, dilation=1, groups=1):
    x, ppm, bg = ctx.saved_variables
    x_grad = w_grad = bias_grad = None
    if ctx.needs_input_grad[0]:
      x_grad = pwm_conv1d_input(x, ppm, bg, grad_output, stride, padding, dilation, groups)
    if ctx.needs_input_grad[1]:
      w_grad = pwm_conv1d_weight(x, ppm, grad_output, stride, padding, dilation, groups)
    if ctx.needs_input_grad[3]:
      bias_grad = pwm_conv1d_bias(x, ppm, grad_output, stride, padding, dilation, groups)
    return x_grad, w_grad, None, bias_grad, None, None, None, None


class deNovo_Conv1d(nn.Conv1d):
  """
    1d convolutional filter layer in which weights are restricted to be postion weight values (i.e. log-odds).
    Parameters
    ----------
    in_channels : int
        Number of input channels (nucleotides)
    out_channels : int
        Number of filters
    kernel_size : int
        Filter size
    bg_probs : dict
        Dictionary containing probability of seeing a given nucleotide. 
        {'A':0.25,'C':0.25,'G':0.25,'T':0.25} is default.

    """
    def __init__(self, in_channels, out_channels, kernel_size, bg_probs={'A':0.25,'C':0.25,'G':0.25,'T':0.25},
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
      super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

      bg_array = [ v for v in bg_probs.values()]
      bg_matrix = np.repeat(np.reshape(np.array(bg_array),(-1,1)),kernel_size,axis=1)
      bg_tensor = np.repeat(np.reshape(bg_matrix,(1,4,kernel_size)),out_channels,axis=0)
      self.bg_tensor = torch.from_numpy(bg_tensor)
      if torch.cuda.is_available():
        self.bg_tensor = self.bg_tensor.cuda()

    def forward(self, input):
      return pwmConv().apply(input, self.weight, self.bg_tensor, self.bias, self.stride, self.padding, self.dilation, self.groups)

class jaspar_Conv1d(nn.Conv1d):
  """
    1d convolutional filter layer in which weights are restricted to be postion weight values (i.e. log-odds) and specified
    using annotated Jaspar motifs.
    Parameters
    ----------
    annotated_motifs_file : str
        File path to Jaspar-formatted PFMs.
    bg_probs : dict
        Dictionary containing probability of seeing a given nucleotide. 
        {'A':0.25,'C':0.25,'G':0.25,'T':0.25} is default.
    pseudocount : real
        Small value to add to zero-probability values.
    trainable : bool
        Whether filters should be trainable (True) or fixed at annotations (False)

    """
    def __init__(self, annotated_motifs_file, 
                 bg_probs={'A':0.25,'C':0.25,'G':0.25,'T':0.25}, 
                 pseudocount=.01, trainable=False,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

      self.bg_probs = [ v for v in bg_probs.values()]
      self.trainable = trainable
      # Load JASPAR annotated motifs
      filters = []
      names = []
      with open(annotated_motifs_file) as handle: 
        for m in motifs.parse(handle, "jaspar"):
          if self.trainable:
            d_pwm = m.counts.normalize(pseudocounts=pseudocount)
          else:
            d_pwm = m.counts.normalize(pseudocounts=pseudocount).log_odds(bg_probs)
          filters.append(np.matrix([d_pwm['A'], d_pwm['C'],d_pwm['G'],d_pwm['T']]))
          names.append(m.name)
      
      # Pad shorter motifs
      max_len = max([f.shape[1] for f in filters])
      for i in range(len(filters)): 
        if filters[i].shape[1] < max_len:
          needed_buffer = (max_len - filters[i].shape[1])
          front_buffer = np.ceil(needed_buffer/2)
          back_buffer = (needed_buffer-front_buffer)
          filters[i] = np.column_stack((np.zeros((4,front_buffer.astype(np.int64))),filters[i],np.zeros((4,back_buffer.astype(np.int64)))))

      super().__init__(4, len(filters), max_len, stride, padding, dilation, bias=bias)

      if self.trainable:
        bg_array = [ v for v in bg_probs.values()]
        bg_matrix = np.repeat(np.reshape(np.array(bg_array),(-1,1)),max_len,axis=1)
        bg_tensor = np.repeat(np.reshape(bg_matrix,(1,4,max_len)),len(filters),axis=0)
        self.bg_tensor = torch.from_numpy(bg_tensor)
        if torch.cuda.is_available():
          self.bg_tensor = self.bg_tensor.cuda()

      # Initialize weights to motifs
      for i in range(len(filters)):
        self.weight[i] = torch.from_numpy(filters[i])

      # Fix weights to annotated motifs if desired
      if not trainable: 
        for param in self.parameters(): 
          if len(param.shape) > 1:
            param.requires_grad = False

    def forward(self, input):
      if self.trainable:
        return pwmConv().apply(input, self.weight, self.bg_tensor, self.bias, self.stride, self.padding, self.dilation, self.groups)
      else:
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

