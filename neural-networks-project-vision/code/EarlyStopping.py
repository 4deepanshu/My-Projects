import torch


class EarlyStopping:
    """Provides early stopping regularization for neural network training

    Parameters
    ----------
    patience : int, optional
        The number of validation loss checks to wait for an improvement, by default 5
    path : str, optional
        File name to save the state dictionary of the network, by default "checkpoint.pt"

    Attributes
    ----------
    patience : int
        The number of validation loss checks to wait for an improvement
    path : str
        File name to save the state dictionary of the network
    patience_counter : int
        Counts how many times the validation loss did not improve
    min_loss : float
        Smallest so far observed validation loss
    """
    def __init__(self, patience=5, path="checkpoint.pt"):
        self.patience = patience
        self.path = path
        self.patience_counter = 0
        self.min_loss = float("inf")
    
    def check_loss(self, validation_loss, model):
        """Takes note of the current loss and compares it against smallest validation loss

        If an improvement was made the model's state is saved.

        Parameters
        ----------
        validation_loss : float
            Current model's loss on the validation set
        model : torch.nn.Module
            Currently trained neural network model
        """
        if validation_loss < self.min_loss:
            torch.save(model.state_dict(), self.path)
            self.min_loss = validation_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def check_stop(self):
        """Returns whether training should stop (patience was reached)
        """
        return self.patience_counter >= self.patience
