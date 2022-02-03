from torchvision.datasets import Cityscapes
import torchvision.transforms as trans
import torch
from torch.utils.data import random_split
from cityscapesscripts.helpers import labels

class CityscapesLoader:
    """Loads and pre-processes the CityScapes dataset for the experiments

    Parameters
    ----------
    dataRoot : str, optional
        Directory where the dataset was downloaded and extracted to, by default "./data/cityscapes"
    imageSize : tuple, optional
        Size to scale images and segmentation maps to as (height, width), by default (256, 512)
    valSetSize : int, optional
        Number of samples in the custom validation set that is split apart from the training set, by default 300

    Attributes
    ----------
    dataRoot : str
        Directory where the dataset was downloaded and extracted to
    imageSize : tuple
        Size to scale images and segmentation maps to as (height, width)
    valSetSize : int
        Number of samples in the custom validation set that is split apart from the training set
    """
    
    def __init__(self, dataRoot="./data/cityscapes", imageSize=(256, 512), valSetSize=300):
        self.dataRoot = dataRoot
        self.imageSize = imageSize
        self.valSetSize = valSetSize
        
    def load(self):
        """Loads and preprocesses the dataset

        For preprocessing all images are resized and converted to tensors. The segmentation maps are multiplied to
        have integer class labels. Class labels are converted from all ids to only the training IDs that are typically
        used for Cityscapes evaluation. Unnecessary dimensions in the feature map tensors are removed.

        Also, since the original test set basically has no labels, the original validation set is used as the test set.
        The original training set is randomly split into training and validation set.

        Returns
        -------
        tuple
            training set, validation set, test set
        """
        dataQualityMode = "fine"
        dataTargetType = "semantic"
        transform = trans.Compose([
            trans.Resize(self.imageSize),
            trans.ToTensor()
        ])
        target_transform = trans.Compose([
            trans.Resize(self.imageSize, interpolation=0), # prevent interpolation between classes
            trans.ToTensor(),
            lambda x: (255 * x).long(), # correct class labels to be integers
            lambda x: x.squeeze(), # remove unnecessary dimensions
            self.convert_to_train_id # only use the train IDs
        ])
        
        # use validation set as test set since actual test set has no labels
        testSet = Cityscapes(self.dataRoot, "val", dataQualityMode, dataTargetType, transform, target_transform)
        # split training set to still have a validation set for early stopping and model selection
        completeTrainSet = Cityscapes(self.dataRoot, "train", dataQualityMode, dataTargetType, transform, target_transform)
        trainSet, valSet = random_split(completeTrainSet,
                                        lengths=[len(completeTrainSet)-self.valSetSize, self.valSetSize])
        
        return trainSet, valSet, testSet
    
    def convert_to_train_id(self, ids):
        """Converts class IDs into the more coarse-grained training IDs used for Cityscapes evaluation

        Parameters
        ----------
        ids : torch.tensor
            Tensor containing valid Cityscapes class labels

        Returns
        -------
        torch.tensor
            Tensor of the same shape as ids where all IDs were replaced by training IDs
        """
        for label in labels.labels:
            ids[ids == label.id] = label.trainId
        return ids
