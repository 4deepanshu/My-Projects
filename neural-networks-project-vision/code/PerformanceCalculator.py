import sklearn.metrics as m
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class PerformanceCalculator:
    """Calculates classification performance metrics for neural networks
    """
    def compute_model_outputs(self, model, dataset, batch_size, device):
        """Computes the outputs of a model for the given dataset

        The computation is executed batch-wise on the given device.
        Moreover the model is put into evaluation mode and gradients are turned off.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network to compute outputs for
        dataset : torch.utils.data.dataset.Dataset
            The data to compute model outputs for
        batch_size : int
            The number of samples to compute the outputs for simultaneously on the device
        device : torch.device
            The device (e.g. GPU) to compute the outputs on

        Returns
        -------
        torch.tensor
            Output tensor of dimensionality specified by the model and the dataset
        """
        data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=4)
        train_mode = model.training
        model.train(False)
        outputs = []
        with torch.no_grad():
            for xi, _ in data_loader:
                # move data to device
                xi = xi.to(device)
                # forward pass through model
                output = model(xi)
                outputs.append(output.cpu())
        model.train(train_mode)
        return torch.cat(outputs)
    
    def compute_predictions(self, model_outputs):
        """Compute a model's class predictions from its outputs

        Performs an argmax across dimension 1.

        Parameters
        ----------
        model_outputs : torch.tensor
            Model output tensor where dimension 1 represents the possible classes

        Returns
        -------
        torch.tensor
            Model output tensor where dimension 1 has been collapsed by argmax
        """
        return torch.argmax(model_outputs, dim=1)
    
    def filter_labeled_pixels(self, y, y_true, non_label=255):
        """Filters for outputs whose true class is not the non_label

        Parameters
        ----------
        y : torch.tensor
            Output tensor to filter
        y_true : torch.tensor
            True output tensor of the same shape as y where some non_labels exist
        non_label : int, optional
            The label in y_true to be considered as a NULL label, by default 255

        Returns
        -------
        torch.tensor
            Flattened version of y without the outputs where y_true was the non_label
        """
        return y[y_true != non_label]
    
    def compute_confusion_matrix(self, y_pred, y_true, pos_label):
        """Computes the confusion matrix for the given class

        Parameters
        ----------
        y_pred : torch.tensor
            One-dimensional tensor with all class labels predicted by the model
        y_true : torch.tensor
            Tensor of same shape as y_pred with all ground-truth class labels
        pos_label : int
            The "true" class label that exists in y_pred and y_true to compute the confusion matrix for

        Returns
        -------
        tuple of int
            Tuple with number of (true positives, true negatives, false positives, false negatives)
        """
        pos = y_true == pos_label
        neg = y_true != pos_label
        predicted_pos = y_pred == pos_label
        predicted_neg = y_pred != pos_label
        
        tp = torch.sum(predicted_pos[pos]).item()
        tn = torch.sum(predicted_neg[neg]).item()
        fp = torch.sum(predicted_pos[neg]).item()
        fn = torch.sum(predicted_neg[pos]).item()
        
        assert tp + tn + fp + fn == len(y_pred)
        assert tp + tn + fp + fn == len(y_true)

        return tp, tn, fp, fn
    
    def score_per_label(self, y_pred, y_true, metric):
        """Computes performance scores per existing class label

        Parameters
        ----------
        y_pred : torch.tensor
            One-dimensional tensor with all class labels predicted by the model
        y_true : torch.tensor
            Tensor of same shape as y_pred with all ground-truth class labels
        metric : str
            The metric to compute, one of: accuracy, sensitivity, specificity, dice, jaccard, f1

        Returns
        -------
        dict
            Dictionary with all unique class labels that exist in y_true as keys and the
            performance metric for each class as values
        """
        if metric == "accuracy":
            scorer = lambda tp, tn, fp, fn: (tp + tn) / (tp + tn + fp + fn)
        elif metric == "sensitivity":
            scorer = lambda tp, tn, fp, fn: tp / (tp + fn)
        elif metric == "specificity":
            scorer = lambda tp, tn, fp, fn: tn / (tn + fp)
        elif metric == "dice" or metric == "f1":
            scorer = lambda tp, tn, fp, fn: 2 * tp / (2 * tp + fp + fn)
        elif metric == "jaccard":
            scorer = lambda tp, tn, fp, fn: tp / (tp + fp + fn)
        else:
            raise "Invalid metric"
        
        existing_labels = torch.unique(y_true)
        class_scores = {}
        for label in existing_labels:
            tp, tn, fp, fn = self.compute_confusion_matrix(y_pred, y_true, label)
            class_scores[label.item()] = scorer(tp, tn, fp, fn)
        return class_scores
    
    def mean_score(self, score_per_label):
        """Computes the unweighted mean score across classes

        Parameters
        ----------
        score_per_label : dict
            Dictionary with existing class labels as keys and the
            performance metric for each class as values

        Returns
        -------
        float
            Unweighted mean score across classes
        """
        return np.mean([value for key, value in score_per_label.items()])
