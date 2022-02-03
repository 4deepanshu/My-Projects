import sklearn.metrics as m
import pandas as pd
import torch
import jsonpickle
from PerformanceCalculator import PerformanceCalculator
from CityscapesLoader import CityscapesLoader
from torch.utils.data import DataLoader
from cityscapesscripts.helpers import labels


class PerformanceLogger:
    """Creates a CSV file with test-set performance metrics from a training log

    Parameters
    ----------
    device : torch.device
        The device (e.g. GPU) to compute the model predictions on
    verbose : bool, optional
        Whether to print status information about ongoing operations to the console, by default True

    Attributes
    ----------
    device : torch.device
        The device (e.g. GPU) to compute the model predictions on
    verbose : bool
        Whether to print status information about ongoing operations to the console
    per_class_metrics : list of str
        The performance metrics that are calculated for each class
    overall_metrics : list of str
        The performance metrics that are calculated for the dataset in total (includes per_class_metrics as averages)
    """
    def __init__(self, device, verbose=True):
        self.device = device
        self.verbose = verbose
        self.per_class_metrics = ["sensitivity", "specificity", "dice", "jaccard"]
        self.overall_metrics = self.per_class_metrics + ["accuracy"]
        
    def create_log(self, path, timestamp, decimals=4):
        """Creates a CSV file with test-set performance metrics from a training log

        The CSV file name is composed of the given timestamp and "-eval.csv".

        Parameters
        ----------
        path : str
            Directory where the training log is stored and the CSV file will be stored
        timestamp : str
            The timestamp part of the training log to compute performance metrics for
        decimals : int, optional
            The number of decimals to round all performance metrics to, by default 4
        """
        # load training log and dataset with the same split as during training
        model, train_logger = self.load_training_log(path, timestamp)
        torch.manual_seed(train_logger.random_seed)
        loader = CityscapesLoader()
        _, _, testSet = loader.load()
        
        # calculate class predictions of the model
        if self.verbose:
            print("Calculating model predictions...")
        calc = PerformanceCalculator()
        outputs = calc.compute_model_outputs(model, testSet,
                                             train_logger.experiment_configuration.val_batch_size, self.device)
        y_pred = calc.compute_predictions(outputs)

        # load the true labels from the dataset
        if self.verbose:
            print("Loading true labels...")
        _, y_true = next(iter(DataLoader(testSet, batch_size=len(testSet), shuffle=False, num_workers=4)))
        
        # filter labels for labeled ones (ignore unlabeled pixels)
        y_pred = calc.filter_labeled_pixels(y_pred, y_true)
        y_true = calc.filter_labeled_pixels(y_true, y_true)
        
        # build row index of CSV from all existing class names and the totals
        existing_labels = torch.unique(y_true).numpy()
        index = [labels.trainId2label[label].name for label in existing_labels] + ["total"]
        
        # create dataframe to store performance metrics
        frame = pd.DataFrame(index=index, columns=self.overall_metrics, dtype="float64")
        
        for metric in self.per_class_metrics:
            # calculate all per-class-metrics
            if self.verbose:
                print("Calculating", metric, "...")
            scores = calc.score_per_label(y_pred, y_true, metric)
            # save per-class and mean score in dataframe
            totalScore = calc.mean_score(scores)
            frame[metric] = [val for key, val in scores.items()] + [totalScore]
        
        # calculate accuracy over the total dataset
        frame.loc["total", "accuracy"] = m.accuracy_score(y_true, y_pred)
        
        # round metrics and save CSV
        frame = frame.round(decimals=decimals)
        frame.to_csv(path + timestamp + "-eval.csv")
        
    def load_training_log(self, path, timestamp):
        """Loads the training log consisting of the ExperimentLogger and the final model

        Parameters
        ----------
        path : str
            Directory where the training log is stored
        timestamp : str
            The timestamp part of the training log to load

        Returns
        -------
        tuple (torch.nn.Module, ExperimentLogger)
            The neural network and the training/experiment log
        """
        model = torch.load(path + timestamp + "-model.pt")
        with open(path + timestamp + "-log.json", "r") as file:
            train_logger = jsonpickle.decode(file.read())
        return model, train_logger
