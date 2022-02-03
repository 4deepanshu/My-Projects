class UNetUtil:
    """Utility functions for implementations of U-Net variants"""
    def expandChannels(self, channels):
        """Duplicates the second to last element in the channels list

        This is needed to make layer creation easier since the last RecurrentResidualUnit does not cut down the
        channels to the output channels but rather outputs the second to last number of channels.

        Parameters
        ----------
        channels : list of int
            The network achitecture in terms of the number of channels that each part of the network should have.
            The list must have an odd number of elements. Except for the first and last element
            (input and output channels) the elements need to be symmetric i.e.
            channels[k] == channels[-(k+1)] for k=1,2,...,len(channels)-2

        Returns
        -------
        list of int
            The input channel list just with the second to last elment twice.
            e.g. [3, 16, 32, 64, 128, 64, 32, 16, 1] -> [3, 16, 32, 64, 128, 64, 32, 16, 16, 1]
        """
        expandedChannels = [(c,) if i != len(channels) - 2 else (c,c) for i, c in enumerate(channels)]
        expandedChannels = [c for tup in expandedChannels for c in tup]
        return expandedChannels

    def add_module_list(self, network, modules, name):
        """Makes list of modules be submodules of the network

        Parameters
        ----------
        network : torch.nn.Module
            Parent module that the list of modules should be part of
        modules : list of torch.nn.Module
            list of submodules to add to the network
        name : str
            Prefix used to name the submodules inside the network
        """
        for i, module in enumerate(modules):
            network.add_module(name + str(i), module)
