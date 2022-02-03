import torch
import torch.nn as nn
from Conv2dLSTM import Conv2dLSTM

class BiConvLSTM(nn.Module):
    """Bi-Directional Convolutional Long-short Term Memory (BiConvLSTM) Block

    Combines spatial and temporal information of Conv2dLSTMs in the forward and backward temporal direction.

    Parameters
    ----------
    in_channels : int
        The number of channels to be input into this block
    out_channels : int
        The number of channels to be output by this block

    Attributes
    ----------
    tanh : torch.nn.Tanh
        Hyperbolic tangent activation function
    forward_lstm : Conv2dLSTM
        Convolutional LSTM used to compute the data in the forward temporal direction
    backward_lstm : Conv2dLSTM
        Convolutional LSTM used to compute the data in the backward temporal direction
    conv_f : torch.nn.Conv2d
        Convolutional layer used to combine the forward_lstm outputs to the final output
    conv_b : torch.nn.Conv2d
        Convolutional layer used to combine the backward_lstm outputs to the final output
    """
    def __init__(self, in_channels, out_channels):
        super(BiConvLSTM, self).__init__()
        self.tanh = nn.Tanh()
        kernel_size = 3
        stride = 1
        padding = 1
        self.add_module("forward_lstm", Conv2dLSTM(in_channels, out_channels))
        self.add_module("backward_lstm", Conv2dLSTM(in_channels, out_channels))
        self.add_module("conv_f", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_b", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        """Computes the forward pass through the block

        Parameters
        ----------
        x : torch.tensor
            Block input with shape (timesteps, batch_size, in_channels, height, width)

        Returns
        -------
        torch.tensor
            Block output with shape (timesteps, batch_size, out_channels, height, width)
        """
        # get the input tensor in reversed temporal order
        timesteps = len(x)
        reverse_time_index = torch.tensor(list(range(timesteps-1, -1, -1)), dtype=torch.int64, device=x.device)
        reverse_x = torch.index_select(x, dim=0, index=reverse_time_index)
        
        # pass the input through both LSTMs
        hf = self.forward_lstm(x)
        hb = self.backward_lstm(reverse_x)
        
        # combine LSTM outputs of both temporal directions
        outputs = []
        for i in range(timesteps):
            out = self.tanh(self.conv_f(hf[i]) + self.conv_b(hb[-(i+1)]))
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)
