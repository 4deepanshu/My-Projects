import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from EarlyStopping import EarlyStopping
from LossCalculator import LossCalculator


class ModelTrainer:
    """Trains a neural network model with early stopping while logging training and vaidation losses between epochs

    Parameters
    ----------
    train_set : torch.utils.data.dataset.Dataset
        The dataset to train the model with
    val_set : torch.utils.data.dataset.Dataset
        The dataset to calculate the validation loss with that is used for early stopping
    model : torch.nn.Module 
        Neural network to train
    device : torch.device
        The device (e.g. GPU) to train and validate the model on
    loss_function : loss function from torch.nn
        The loss function to compute the loss with
    optimizer : torch.optim.Optimizer
        The optimizer used to take gradient-based optimization steps
    logger : ExperimentLogger
        The logger used to save training and validation losses between epochs
    patience : int, optional
        The number of epochs to wait for a validation loss improvement before the training is stopped, by default 5
    train_batch_size : int, optional
        The number of samples to compute the backpropagation (training) loss for simultaneously on the device, by default 16
    val_batch_size : int, optional
        The number of samples to compute the logging (training and validation) loss for simultaneously on the device, by default 64
    verbose : bool, optional
        Whether to print messages about epoch/batch progress and loss calculation to the console, by default True

    Attributes
    ----------
    train_set : torch.utils.data.dataset.Dataset
        The dataset to train the model with
    val_set : torch.utils.data.dataset.Dataset
        The dataset to calculate the validation loss with that is used for early stopping
    model : torch.nn.Module 
        Neural network to train
    device : torch.device
        The device (e.g. GPU) to train and validate the model on
    loss_function : loss function from torch.nn
        The loss function to compute the loss with
    checkpoint_file : str
        File name used to store early stopping checkpoints
    optimizer : torch.optim.Optimizer
        The optimizer used to take gradient-based optimization steps
    logger : ExperimentLogger
        The logger used to save training and validation losses between epochs
    verbose : bool, optional
        Whether to print messages about epoch/batch progress and loss calculation to the console, by default True
    train_loader : torch.utils.data.DataLoader
        Data loader used to obtain samples for training
    early_stop : EarlyStopping
        Early stopping object used to track best validation loss, save checkpoints and stop training
    train_calculator : LossCalculator
        Calculator to compute training loss between epochs
    val_calculator : LossCalculator
        Calculator to compute validation loss between epochs
    """
    def __init__(self, train_set, val_set, model, device, loss_function, optimizer, logger, patience=5,
                 train_batch_size=16, val_batch_size=64, verbose=True):
        self.train_set = train_set
        self.val_set = val_set
        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.checkpoint_file = "checkpoint.pt"
        self.optimizer = optimizer
        self.logger = logger
        self.verbose = verbose
        
        self.train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4)
        self.early_stop = EarlyStopping(patience, self.checkpoint_file)
        
        self.train_calculator = LossCalculator(train_set, loss_function, val_batch_size, device)
        self.val_calculator = LossCalculator(val_set, loss_function, val_batch_size, device)
    
    def train(self, epochs):
        """Trains the model batch-wise with early stopping and while logging training and validation losses

        Parameters
        ----------
        epochs : int
            The maximum number of training iterations (passes through the whole training set)
        """
        # log initial losses
        self.log_losses(self.model)
        
        for epoch in range(epochs):
            
            if self.verbose:
                print("Epoch", epoch + 1)
                
            for i, (xi, yi) in enumerate(self.train_loader):
                # move data to device
                xi = xi.to(self.device)
                yi = yi.to(self.device)
                # reset previous gradients
                self.optimizer.zero_grad()
                # forward pass through model
                output = self.model(xi)
                # calculate current loss of model
                loss = self.loss_function(output, yi)
                # backprop
                loss.backward()
                # take optimization step
                self.optimizer.step()
                if self.verbose:
                    print(".", end="")
                    
            if self.verbose:
                print()
            
            # log losses after epoch
            self.log_losses(self.model)
            
            # check for and log early stopping
            if self.early_stop.check_stop():
                self.logger.set_stopped_early(True)
                break
        
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        
    def log_losses(self, model):
        """Computes and adds the current model's losses to the logger attribute
        
        Also updates the early stopping attribute with the current validation loss.

        Parameters
        ----------
        model : torch.nn.Module
            The current model
        """
        if self.verbose:
            print("Computing training loss...")
        train_loss = self.train_calculator.compute_loss(model)
        
        if self.verbose:
            print("Computing validation loss...")
        val_loss = self.val_calculator.compute_loss(model)
        
        self.logger.log_validation_loss(val_loss)
        self.logger.log_training_loss(train_loss)
        
        self.early_stop.check_loss(val_loss, model)
