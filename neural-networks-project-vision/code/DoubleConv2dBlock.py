import torch.nn as nn

class DoubleConv2dBlock(nn.Module):
    """Block of two consecutive convolutional layers with a ReLU activation function

    Parameters
    ----------
    in_channels : int
        The number of channels to be input into this block
    out_channels : int
        The number of channels to be output by this block
    kernel_size : int, optional
        Kernel size of the 2D convolutional layer, by default 3
    stride : int, optional
        Stride of the 2D convolutional layer, by default 1
    padding : int, optional
        Padding of the 2D convolutional layer, by default 1
    **kwargs
        Further arguments to be passed to the 2D convolutional layer

    Attributes
    ----------
    cnn1 : torch.nn.Conv2d
        First convolutional layer mapping from in_channels to out_channels
    cnn2: torch.nn.Conv2d
        Second convolutional layer mapping from out_channels to out_channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super(DoubleConv2dBlock, self).__init__()
        self.relu = nn.ReLU()
        self.add_module("cnn1", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs))
        self.add_module("cnn2", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, **kwargs))

    def forward(self, x):
        """Computes the forward pass through the block

        Parameters
        ----------
        x : torch.tensor
            Block input with shape (batch_size, in_channels, height, width)

        Returns
        -------
        torch.tensor
            Block output with shape (batch_size, out_channels, height, width) if the default kernel_size, stride and padding are used
        """
        out = self.relu(self.cnn1(x))
        out = self.relu(self.cnn2(out))
        return out

