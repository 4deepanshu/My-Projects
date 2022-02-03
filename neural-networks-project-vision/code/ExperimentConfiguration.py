from ClassConfiguration import ClassConfiguration


class ExperimentConfiguration:
    """Holds information about how a training experiment was configured

    Parameters
    ----------
    model_conf : ClassConfiguration
        The configuration of the neural network model (type and arguments)
    optimizer_conf : ClassConfiguration
        The configuration of the optimizer (type and arguments)
    loss_conf : ClassConfiguration
        The configuration of the loss function (type and arguments)
    epochs : int
        The number of training iterations through the training set
    train_batch_size : int
        The batch size used to determine optimization steps during training
    val_batch_size : int
        The batch size used to compute losses for logging (no backpropagation)
    early_stop_patience : int
        The number of epochs to wait for validation loss to decrease before
        the training is stopped.

    Attributes
    ----------
    model_conf : ClassConfiguration
        The configuration of the neural network model (type and arguments)
    optimizer_conf : ClassConfiguration
        The configuration of the optimizer (type and arguments)
    loss_conf : ClassConfiguration
        The configuration of the loss function (type and arguments)
    epochs : int
        The number of training iterations through the training set
    train_batch_size : int
        The batch size used to determine optimization steps during training
    val_batch_size : int
        The batch size used to compute losses for logging (no backpropagation)
    early_stop_patience : int
        The number of epochs to wait for validation loss to decrease before
        the training is stopped.
    """
    def __init__(self, model_conf, optimizer_conf, loss_conf,
                 epochs, train_batch_size, val_batch_size, early_stop_patience):
        self.model_conf = model_conf
        self.optimizer_conf = optimizer_conf
        self.loss_conf = loss_conf
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.early_stop_patience = early_stop_patience
