import torch
from torch.utils.data import DataLoader


class LossCalculator:
    """Calculates the loss of a model for logging (no gradients)

    Parameters
    ----------
    dataset : torch.utils.data.dataset.Dataset
        The data to compute the loss on
    loss_function : loss function from torch.nn
        The loss function to compute the loss with
    batch_size : int
        The number of samples to compute the loss for simultaneously on the device
    device : torch.device
        The device (e.g. GPU) to compute the loss on

    Attributes
    ----------
    dataset : torch.utils.data.dataset.Dataset
        The data to compute the loss on
    data_loader : torch.utils.data.DataLoader
        The data loader used to compute the loss batch-wise
    loss_function : loss function from torch.nn
        The loss function to compute the loss with
    device : torch.device
        The device (e.g. GPU) to compute the loss on
    """
    def __init__(self, dataset, loss_function, batch_size, device):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=4)
        self.device = device
        self.loss_function = loss_function
        
    def compute_loss(self, model):
        """Computes the loss of the model in training mode without gradients

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to compute the loss for

        Returns
        -------
        float
            The current average loss of the model across all dataset samples
        """
        train_mode = model.training
        model.train(False)
        with torch.no_grad():
            weightedLossSum = 0
            for xi, yi in self.data_loader:
                # move data to device
                xi = xi.to(self.device)
                yi = yi.to(self.device)
                # forward pass through model
                output = model(xi)
                # calculate current loss of model
                loss = self.loss_function(output, yi)
                weightedLossSum += loss.cpu().item() * len(yi)
        model.train(train_mode)
        
        return weightedLossSum / len(self.dataset)
