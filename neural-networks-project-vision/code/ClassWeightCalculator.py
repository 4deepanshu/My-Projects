import torch
from torch.utils.data import DataLoader


class ClassWeightCalculator:
    """Computes the weights for classes of an imbalanced dataset to be used in torch.nn.CrossEntropyLoss

    Parameters
    ----------
    verbose : bool, optional
        Whether to print dots after each completed batch, by default True

    Attributes
    ----------
    verbose : bool
        Whether to print dots after each completed batch
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def calculate_weights(self, dataset, batch_size, non_label=255):
        """Calculates weights for an imbalanced dataset for torch.nn.CrossEntropyLoss

        The class frequency is computed batch-wise and the weights are to be matched
        with all class labels that exist in the dataset in increasing order. Each class'
        weight is the fraction of the largest class' frequency and its frequency.

        Parameters
        ----------
        dataset : torch.utils.data.dataset.Dataset
            The data with class imbalance to calculate the weights for
        batch_size : int
            The number of samples to count the class frequencies for at a time
        non_label : int, optional
            The class label value that indicates a to-be-ignored class, by default 255

        Returns
        -------
        torch.tensor
            The weights to be passed to torch.nn.CrossEntropyLoss
        """
        data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=4)
        class_frequencies = {}
        
        for _, y in data_loader:
            # get unique labels and their counts for the batch
            existing_labels, counts = torch.unique(y, return_counts=True)
            for label, count in zip(existing_labels, counts):
                if label == non_label: # skip if label should be ignored
                    continue
                # add count of label
                if label in class_frequencies:
                    class_frequencies[label.item()] += count.item()
                else:
                    class_frequencies[label.item()] = count.item()
            if self.verbose:
                print(".", end="")
        if self.verbose:
            print()
        
        # get frequency of predominant class
        largest_class_frequency = class_frequencies[max(class_frequencies, key=lambda x: class_frequencies[x])]
        weights = []
        for label, frequency in sorted(class_frequencies.items()):
            # calculate each class' weight
            weights.append(largest_class_frequency / frequency)
        
        return torch.tensor(weights)
