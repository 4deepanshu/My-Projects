import torch.nn as nn

class RecurrentConv2d(nn.Module):
    """2D convolutional layer that has a feed-forward and simple recurrent connections

    Parameters
    ----------
    in_channels : int
        The number of channels to be input into this layer
    out_channels : int
        The number of channels to be output by this layer
    kernel_size : int, optional
        Kernel size of the 2D convolutional layer, by default 3
    stride : int, optional
        Stride of the 2D convolutional layer, by default 1
    padding : int, optional
        Padding of the 2D convolutional layer, by default 1
    timesteps : int, optional
        How many times the recurrent layer will be applied to the feed-forward output, by default 2
    **kwargs
        Further arguments to be passed to the 2D convolutional layer

    Attributes
    ----------
    timesteps : int
        How many times the recurrent layer will be applied to the feed-forward output
    feed_forward : torch.nn.Conv2d
        2D convolutional layer for the feed-forward connection
    recurrent : torch.nn.Conv2d
        2D convolutional layer for the recurrent connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, timesteps=2, **kwargs):
        super(RecurrentConv2d, self).__init__()
        self.timesteps = timesteps
        self.add_module("feed_forward", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs))
        self.add_module("recurrent", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, **kwargs))
        
    def forward(self, x):
        """Computes the forward pass through the layer

        Parameters
        ----------
        x : torch.tensor
            Layer input with shape (batch_size, in_channels, height, width)

        Returns
        -------
        torch.tensor
            Layer output with shape (batch_size, out_channels, height, width) if the default kernel_size, stride and padding are used
        """
        out_forward = self.feed_forward(x)
        out = out_forward
        for i in range(self.timesteps):
            out = self.recurrent(out) + out_forward
        return out
