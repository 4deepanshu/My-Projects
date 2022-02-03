import torch
import torch.nn as nn
from RecurrentResidualUnit import RecurrentResidualUnit
from UNetUtil import UNetUtil

class R2UNet(nn.Module):
    """Recurrent Residual Convolutional Neural Network based on U-Net

    Parameters
    ----------
    channels : list of int, by default [3, 16, 32, 64, 128, 64, 32, 16, 1]
        The network achitecture in terms of the number of channels that each part of the network should have.
        The list must have an odd number of elements. Except for the first and last element
        (input and output channels) the elements need to be symmetric i.e.
        channels[k] == channels[-(k+1)] for k=1,2,...,len(channels)-2
    kernel_size : int, optional
        Kernel size of the 2D convolutional layer, by default 3
    stride : int, optional
        Stride of the 2D convolutional layer, by default 1
    padding : int, optional
        Padding of the 2D convolutional layer, by default 1
    timesteps : int, optional
        timesteps to be passed to the RecurrentResidualUnit layers, by default 2
    **kwargs
        Further arguments to be passed to the RecurrentResidualUnit layers

    Attributes
    ----------
    pool : torch.nn.MaxPool2d
        2x2 max-pooling layer
    relu : torch.nn.ReLU
        ReLU activation function
    downsampling_layers : list of RecurrentResidualUnit
        Recurrent residual units used in the encoding part of the network
    upsampling_layers : list of RecurrentResidualUnit
        Recurrent residual units used in the decoding part of the network
    transposed_conv_layers : list of torch.nn.ConvTranspose2d
        Transposed convolutional layers to increase the feature map's spatial dimensions in the decoding part of the network
    downX : RecurrentResidualUnit
        Element of downsampling_layers
    upX : RecurrentResidualUnit
        Element of upsampling_layers
    tconvX : torch.nn.ConvTranspose2d
        Element of transposed_conv_layers
    final_conv_layer : torch.nn.Conv2d
        Final convolutional layer with 1x1 kernel to get the desired number of output channels 
    """
    def __init__(self, channels=[3, 16, 32, 64, 128, 64, 32, 16, 1],
                kernel_size=3, stride=1, padding=1, timesteps=2, **kwargs):
        super(R2UNet, self).__init__()
        util = UNetUtil()
        expandedChannels = util.expandChannels(channels)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.downsampling_layers = []
        self.upsampling_layers = []
        self.transposed_conv_layers = []

        middleChannelIndex = (len(channels) - 1)//2
        for i in range(len(expandedChannels) - 2):
            if i < middleChannelIndex: # encoding part
                # encoding layers map from current to next channels
                down_layer = RecurrentResidualUnit(expandedChannels[i], expandedChannels[i+1], kernel_size, stride, padding, timesteps)
                self.downsampling_layers.append(down_layer)
            else: # decoding part
                # decoding layers map from twice the next channels (twice due to concatenation) to the number of channels after that
                up_layer = RecurrentResidualUnit(2 * expandedChannels[i+1], expandedChannels[i+2], kernel_size, stride, padding, timesteps)
                self.upsampling_layers.append(up_layer)
                # transposed layers keep number of channels but increase spatial dimensions
                tconv_layer = nn.ConvTranspose2d(expandedChannels[i+1], expandedChannels[i+1], 3, 2, 1, 1)
                self.transposed_conv_layers.append(tconv_layer)
                
        # special case for first transposed convolutional layer
        # other transposed layers keep number of channels, this one increases the number of channels
        self.transposed_conv_layers[0] = nn.ConvTranspose2d(channels[middleChannelIndex],
                                                            channels[middleChannelIndex+1], 3, 2, 1, 1)
        
        # add final layer to get output channels
        self.add_module("final_conv_layer", nn.Conv2d(expandedChannels[-2], expandedChannels[-1], 1))
        
        # add all layers as modules
        util.add_module_list(self, self.downsampling_layers, "down")
        util.add_module_list(self, self.upsampling_layers, "up")
        util.add_module_list(self, self.transposed_conv_layers, "tconv")
        
    def forward(self, x):
        """Computes the forward pass through the network

        Parameters
        ----------
        x : torch.tensor
            Network input of shape (batch_size, channels[0], height, width)

        Returns
        -------
        torch.tensor
            Network output with shape (batch_size, channels[-1], height, width) if the default kernel_size, stride and padding are used
        """
        downsampling_outputs = []
        out = x
        # encoding part
        for down_layer in self.downsampling_layers[:-1]:
            out = down_layer(out) # apply encoding layer
            downsampling_outputs.append(out) # save encoding output for decoding part
            out = self.pool(out) # apply max pooling
        out = self.downsampling_layers[-1](out) # last downsampling layer without pooling

        # decodig part
        for tconv_layer, up_layer, down_out in zip(self.transposed_conv_layers, self.upsampling_layers,
                                                   downsampling_outputs[::-1]):
            out = self.relu(tconv_layer(out)) # use transposed layer with ReLU
            out = torch.cat([out, down_out], dim=1) # concatenate output of encoding layer
            out = up_layer(out) # apply decoding layer
        out = self.final_conv_layer(out) # use finaly layer to match channels
        return out
