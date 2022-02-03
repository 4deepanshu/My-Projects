import torch
import torch.nn as nn

class Conv2dLSTM(nn.Module):
    """Long-short Term Memory (LSTM) Cell with 2D convolutional operations

    LSTM that uses convolutional layers to compute gates (input, forget, output) and states
    (cell, hidden) to leverage spatial and sequence information.

    Parameters
    ----------
    in_channels : int
        The number of channels to be input into this cell
    out_channels : int
        The number of channels to be output by this cell, i.e. number of channels for the
        cell state and the hidden state
    
    Attribues
    ----------
    in_channels : int
        The number of channels to be input into this cell
    out_channels : int
        The number of channels to be output by this cell, i.e. number of channels for the
        cell state and the hidden state
    sigmoid : torch.nn.Sigmoid
        Sigmoid activation function
    tanh : torch.nn.Tanh
        Hyperbolic tangent activation function
    conv_xi : torch.nn.Conv2d
        Convolutional layer to map input x to input gate i
    conv_hi : torch.nn.Conv2d
        Convolutional layer to map hidden state h to input gate i
    conv_xf : torch.nn.Conv2d
        Convolutional layer to map input x to forget gate f
    conv_hf : torch.nn.Conv2d
        Convolutional layer to map hidden state h to forget gate f
    conv_xo : torch.nn.Conv2d
        Convolutional layer to map input x to output gate o
    conv_ho : torch.nn.Conv2d
        Convolutional layer to map hidden state h to output gate o
    conv_xc : torch.nn.Conv2d
        Convolutional layer to map input x to cell state c
    conv_hc : torch.nn.Conv2d
        Convolutional layer to map hidden state h to cell state c
    """
    def __init__(self, in_channels, out_channels):
        super(Conv2dLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        kernel_size = 3
        stride = 1
        padding = 1
        self.add_module("conv_xi", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_hi", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_xf", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_hf", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_xo", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_ho", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_xc", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module("conv_hc", nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        """Computes the forward pass through the cell

        Parameters
        ----------
        x : torch.tensor
            Cell input with shape (timesteps, batch_size, in_channels, height, width)

        Returns
        -------
        torch.tensor
            Cell output with shape (timesteps, batch_size, out_channels, height, width)
        """
        # get meaningful names of tensor dimensions
        timesteps = x.shape[0]
        batch_size = x.shape[1]
        height = x.shape[3]
        width = x.shape[4]

        # create initial cell and hidden state with all zeros
        h = torch.zeros((batch_size, self.out_channels, height, width), device=x.device)
        c = torch.zeros((batch_size, self.out_channels, height, width), device=x.device)

        hts = []
        for j in range(timesteps): # loop through timesteps
            i = self.sigmoid(self.conv_xi(x[j]) + self.conv_hi(h)) # compute input gate
            f = self.sigmoid(self.conv_xf(x[j]) + self.conv_hf(h)) # compute forget gate
            o = self.sigmoid(self.conv_xo(x[j]) + self.conv_ho(h)) # compute output gate
            c = f * c + i * self.tanh(self.conv_xc(x[j]) + self.conv_hc(h)) # compute cell state
            h = o * self.tanh(c) # compute hidden state
            hts.append(h) # save hidden state for output
        
        # output hidden states as tensor
        return torch.stack(hts, dim=0)
