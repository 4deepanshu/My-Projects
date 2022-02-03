import torch
import torch.nn as nn
from RecurrentResidualUnit import RecurrentResidualUnit
from DoubleConv2dBlock import DoubleConv2dBlock
from DenseConvBlock import DenseConvBlock
from BiConvLSTM import BiConvLSTM
from UNetUtil import UNetUtil

class BCDUNet(nn.Module):
    """Bi-Directional ConvLSTM U-Net with Densley Connected Convolutions

    Parameters
    ----------
    channels : list of int, by default [3, 16, 32, 64, 128, 64, 32, 16, 1]
        The network achitecture in terms of the number of channels that each part of the network should have.
        The list must have an odd number of elements. Except for the first and last element
        (input and output channels) the elements need to be symmetric i.e.
        channels[k] == channels[-(k+1)] for k=1,2,...,len(channels)-2
    depth : int
        The number of chained DoubleConv2dBlocks in the DenseConvBlock at the bottom of the U
    kernel_size : int, optional
        Kernel size of the 2D convolutional layer, by default 3
    stride : int, optional
        Stride of the 2D convolutional layer, by default 1
    padding : int, optional
        Padding of the 2D convolutional layer, by default 1

    Attributes
    ----------
    pool : torch.nn.MaxPool2d
        2x2 max-pooling layer
    relu : torch.nn.ReLU
        ReLU activation function

    downsampling_layers : list of DoubleConv2dBlock
        Convolutional blocks used in the encoding part of the network
    upsampling_layers : list of DoubleConv2dBlock
        Convolutional blocks used in the decoding part of the network
    transposed_conv_layers : list of torch.nn.ConvTranspose2d
        Transposed convolutional layers to increase the feature map's spatial dimensions and decrease
        its channels in the decoding part of the network
    batch_norm_layers : list of torch.nn.BatchNorm2d
        Batch normalization layers in the decoding part of the network to regularize activations
    lstms : list of BiConvLSTM
        LSTM blocks that combine the encoding outputs and the previous decoding outputs in the decoding
        part of the network
    conv_layers : list of torch.nn.Conv2d
        Convolutional layers used in the decoding part of the network between the lstm and the upsampling_layer

    downX : DoubleConv2dBlock
        Element of downsampling_layers
    upX : DoubleConv2dBlock
        Element of upsampling_layers
    tconvX : torch.nn.ConvTranspose2d
        Element of transposed_conv_layers
    normX : torch.nn.BatchNorm2d
        Element of batch_norm_layers  
    lstmX : BiConvLSTM
        Element of lstms
    convX : torch.nn.Conv2d
        Element of conv_layers

    dense_block : DenseConvBlock
        Dense convolutional block used at the bottom of the U
    final_conv_layer : torch.nn.Conv2d
        Final convolutional layer with 1x1 kernel to get the desired number of output channels 
    """
    def __init__(self, channels=[3, 16, 32, 64, 128, 64, 32, 16, 1], depth=3, kernel_size=3, stride=1, padding=1):
        super(BCDUNet, self).__init__()
        util = UNetUtil()
        expandedChannels = util.expandChannels(channels)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.downsampling_layers = []
        self.upsampling_layers = []
        self.transposed_conv_layers = []
        self.batch_norm_layers = []
        self.lstms = []
        self.conv_layers = []
        
        middleChannelIndex = (len(channels) - 1)//2
        for i in range(len(expandedChannels) - 2):
            if i < middleChannelIndex - 1: # encoding part without dense block at bottom
                # encoding layers map from current to next channels
                down_layer = DoubleConv2dBlock(expandedChannels[i], expandedChannels[i+1], kernel_size, stride, padding)
                self.downsampling_layers.append(down_layer)
            elif i >= middleChannelIndex: # decoding part
                # transposed layers decrease number of channels and increase spatial dimensions
                tconv_layer = nn.ConvTranspose2d(expandedChannels[i], expandedChannels[i+1], 2, 2)
                self.transposed_conv_layers.append(tconv_layer)
                # batch normalization layer for faster convergence
                norm_layer = nn.BatchNorm2d(expandedChannels[i+1])
                self.batch_norm_layers.append(norm_layer)
                # lstm layer combines encoding feature map with feature map from previous decoding layer
                # the output channels are cut in half since re-combine the two timesteps doubles this number again
                lstm = BiConvLSTM(expandedChannels[i+1], expandedChannels[i+1]//2)
                self.lstms.append(lstm)
                # single convolutional layer before double convolutional block
                conv_layer = nn.Conv2d(expandedChannels[i+1], expandedChannels[i+1], kernel_size, stride, padding)
                self.conv_layers.append(conv_layer)
                # decoding layers map from the next channels to the next channels
                up_layer = DoubleConv2dBlock(expandedChannels[i+1], expandedChannels[i+1], kernel_size, stride, padding)
                self.upsampling_layers.append(up_layer)
                
        # add dense block at bottom
        self.add_module("dense_block", DenseConvBlock(channels[middleChannelIndex-1], channels[middleChannelIndex],
                                                      depth, kernel_size, stride, padding))
        
        # add final layer to get output channels
        self.add_module("final_conv_layer", nn.Conv2d(expandedChannels[-2], expandedChannels[-1], 1))
        
        # add all layers as modules
        util.add_module_list(self, self.downsampling_layers, "down")
        util.add_module_list(self, self.upsampling_layers, "up")
        util.add_module_list(self, self.transposed_conv_layers, "tconv")
        util.add_module_list(self, self.batch_norm_layers, "norm")
        util.add_module_list(self, self.lstms, "lstm")
        util.add_module_list(self, self.conv_layers, "conv")
        
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
        for down_layer in self.downsampling_layers:
            out = down_layer(out) # apply encoding layer
            downsampling_outputs.append(out) # save encoding output for decoding part
            out = self.pool(out) # apply max pooling

        out = self.dense_block(out) # dense block at bottom of the U

        # decoding part
        for tconv_layer, norm_layer, up_layer, down_out, lstm_layer, conv_layer \
        in zip(self.transposed_conv_layers, self.upsampling_layers, self.batch_norm_layers,
               downsampling_outputs[::-1], self.lstms, self.conv_layers):
            out = tconv_layer(out) # use transposed layer
            out = norm_layer(out) # normalize activations
            # concatenate outputs of encoding and decoding part as timesteps
            out = torch.stack([out, down_out], dim=0)
            out = lstm_layer(out) # use recurrent layer
            out = out.permute(0, 2, 1, 3, 4) # make first two dimensions the timesteps and the channels
            # collapse timesteps into channels
            out = torch.reshape(out, (out.shape[0] * out.shape[1], out.shape[2], out.shape[3], out.shape[4]))
            out = out.permute(1, 0, 2, 3) # restore order: batch, channel, height, width
            out = self.relu(conv_layer(out)) # apply conv layer that reduces channels
            out = up_layer(out) # apply decoding layer
        out = self.final_conv_layer(out) # use finaly layer to match channels
        return out
