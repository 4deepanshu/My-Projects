from ModelTrainer import ModelTrainer
from CityscapesLoader import CityscapesLoader


class ExperimentRunner:
    """Runs neural network training experiments

    Parameters
    ----------
    logger : ExperimentLogger
        The logger used to save the experiment's configuration, metrics and results
    device : torch.device
        The device (e.g. GPU) to train the model on

    Attributes
    ----------
    logger : ExperimentLogger
        The logger used to save the experiment's configuration, metrics and results
    device : torch.device
        The device (e.g. GPU) to train the model on
    """
    def __init__(self, logger, device):
        self.logger = logger
        self.device = device
        
    def run(self):
        """Runs the training experiment

        The basic steps are: data loading, model creation, model training with loss logging,
        saving the final model with training configuration and losses
        """
        exp_conf = self.logger.experiment_configuration
        # load the data
        loader = CityscapesLoader()
        trainSet, valSet, _ = loader.load()         
        # create the model
        model = exp_conf.model_conf.instance()
        model.to(self.device)
        # train the model
        loss_function = exp_conf.loss_conf.instance()
        optimizer = exp_conf.optimizer_conf.instance(model.parameters())
        trainer = ModelTrainer(trainSet, valSet, model, self.device, loss_function, optimizer,
                               self.logger, exp_conf.early_stop_patience, exp_conf.train_batch_size,
                               exp_conf.val_batch_size)
        trainer.train(exp_conf.epochs)
        # save the results
        self.logger.save_log_file()
        self.logger.save_model(model)
