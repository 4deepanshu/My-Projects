import subprocess
import torch
import jsonpickle
import datetime
import os


class ExperimentLogger:
    """Stores and logs training configuration, training metrics and results

    Parameters
    ----------
    log_path : string
        Diretory to save log files into
    experiment_configuration : ExperimentConfiguration
        Training configuration of the experiment to be logged
    verbose : bool, optional
        Whether to print logged losses during training, by default True

    Attributes
    ----------
    commit_id : string
        Current git commit hash to log the code's version
    random_seed : long
        Initial random seed of pytorch for the experiment
    experiment_configuration : ExperimentConfiguration
        Training configuration of the experiment to be logged
    now : datetime
        The time when the experiment was started
    datetime_format : string
        How to format the timestamps for log file names
    log_path : string
        Diretory to save log files into
    stopped_early : bool
        Whether early stopping ended the training (early)
    verbose : bool
        Whether to print logged losses during training
    validation_losses : list of float
        The logged losses on the validation set between epochs
    training_losses : list of float
        The logged losses on the training set between epochs
    """
    def __init__(self, log_path, experiment_configuration, verbose=True):
        self.commit_id = self.get_commit_id()
        self.random_seed = torch.random.initial_seed()
        self.experiment_configuration = experiment_configuration
        self.now = datetime.datetime.now()
        self.datetime_format = "%Y-%m-%dT%H-%M-%S"
        self.log_path = log_path
        self.stopped_early = False
        self.verbose = verbose
        self.validation_losses = []
        self.training_losses = []
        
    def get_commit_id(self):
        """Returns the current commit ID of the git repository"""
        output = subprocess.check_output(["git", "rev-parse", "HEAD"])
        commit_id = output.decode("ascii").strip()
        return commit_id
    
    def log_validation_loss(self, validation_loss):
        """Adds the given loss to the list of validation losses"""
        self.validation_losses.append(validation_loss)
        if self.verbose:
            print("Validation Loss:", validation_loss)
    
    def log_training_loss(self, training_loss):
        """Adds the given loss to the list of training losses"""
        self.training_losses.append(training_loss)
        if self.verbose:
            print("Training Loss:", training_loss)
        
    def set_stopped_early(self, stopped_early):
        """Sets the stopped_early attribute"""
        self.stopped_early = stopped_early
        
    def save_log_file(self):
        """Saves the ExperimentLogger object to a json file in the log_path

        The file name is the formatted timestamp followed by "-log.json".
        """
        json = jsonpickle.encode(self, indent=4)
        file_name = self.log_path + self.now.strftime(self.datetime_format) + "-log.json"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as file:
            file.write(json)
    
    def save_model(self, model):
        """Saves the given neural network to a pytorch file

        The file name is the formatted timestamp followed by "-model.pt".

        Parameters
        ----------
        model : torch.nn.Module
            The neural network to be saved
        """
        file_name = self.log_path + self.now.strftime(self.datetime_format) + "-model.pt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(model, file_name)
