import torch.nn as nn
from RecurrentConv2d import RecurrentConv2d

class RecurrentResidualUnit(nn.Module):
    """Neural network unit built from 2 RecurrentConv2d layers with ReLU and a residual connection

    Parameters
    ----------
    in_channels : int
        The number of channels to be input into this unit
    out_channels : int
        The number of channels to be output by this unit
    kernel_size : int, optional
        Kernel size of the 2D convolutional layer, by default 3
    stride : int, optional
        Stride of the 2D convolutional layer, by default 1
    padding : int, optional
        Padding of the 2D convolutional layer, by default 1
    timesteps : int, optional
        timesteps to be passed to the RecurrentConv2d layers, by default 2
    **kwargs
        Further arguments to be passed to the RecurrentConv2d layers

    Attributes
    ----------
    relu : torch.nn.ReLU
        ReLU activation function
    rcnn1 : RecurrentConv2d
        first recurrent convolutional layer
    rcnn2 : RecurrentConv2d
        second recurrent convolutional layer
    channel_adaptation : torch.nn.Conv2d
        2D convolutional layer with 1x1 kernel to linearly transform from in_channels to out_channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, timesteps=2, **kwargs):
        super(RecurrentResidualUnit, self).__init__()
        self.relu = nn.ReLU()
        self.add_module("rcnn1", RecurrentConv2d(in_channels, out_channels, kernel_size,
                                                 stride, padding, timesteps, **kwargs))
        self.add_module("rcnn2", RecurrentConv2d(out_channels, out_channels, kernel_size,
                                                 stride, padding, timesteps, **kwargs))
        self.add_module("channel_adaptation", nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        """Computes the forward pass through the unit

        Parameters
        ----------
        x : torch.tensor
            Unit input with shape (batch_size, in_channels, height, width)

        Returns
        -------
        torch.tensor
            Unit output with shape (batch_size, out_channels, height, width) if the default kernel_size, stride and padding are used
        """
        out = self.relu(self.rcnn1(x))
        out = self.relu(self.rcnn2(out))
        out = self.channel_adaptation(x) + out
        return out

