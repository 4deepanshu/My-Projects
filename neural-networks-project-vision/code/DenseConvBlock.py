import torch
import torch.nn as nn
from DoubleConv2dBlock import DoubleConv2dBlock
from UNetUtil import UNetUtil

class DenseConvBlock(nn.Module):
    """Dense convolutional block that chains DoubleConv2dBlocks with skip connections

    For n > 1, the n-th DoubleConv2dBlock gets the concatenated output feature maps of
    all previous DoubleConv2dBlocks as input.

    Parameters
    ----------
    in_channels : int
        The number of channels to be input into this block
    out_channels : int
        The number of channels to be output by this block
    depth : int, optional
        The number of chained DoubleConv2dBlocks, by default 3
    kernel_size : int, optional
        Kernel size of the DoubleConv2dBlock, by default 3
    stride : int, optional
        Stride of the DoubleConv2dBlock, by default 1
    padding : int, optional
        Padding of the DoubleConv2dBlock, by default 1
    **kwargs
        Further arguments to be passed to the DoubleConv2dBlock
    
    Attributes
    ----------
    blocks : list of DoubleConv2dBlock
        The building blocks that this dense block consists of
    blockX : DoubleConv2dBlock
        Element of blocks
    """
    def __init__(self, in_channels, out_channels, depth=3, kernel_size=3, stride=1, padding=1, **kwargs):
        super(DenseConvBlock, self).__init__()
        self.blocks = []
        for i in range(depth):
            if i == 0:
                adapted_channels = in_channels
            else:
                adapted_channels = out_channels * i
            block = DoubleConv2dBlock(adapted_channels, out_channels, kernel_size, stride, padding, **kwargs)
            self.blocks.append(block)
        UNetUtil().add_module_list(self, self.blocks, "block")
    
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
        out = x
        outputs = []
        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block(out)
            else:
                combined_out = torch.cat(outputs, dim=1)
                out = block(combined_out)
            outputs.append(out)
        return out
