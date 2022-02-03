# This script was used to calculate evaluation metrics for trained R2U-Net
# and BCDU-Net models.

import torch
from PerformanceLogger import PerformanceLogger

# only use specific GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"

# specify device depending on availability of GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# specify directory and the file name's timestamp to load the model and
# the training log from
log_path = "../experiments/train_ids/task3_basic_training/"
timestamp = "2021-03-11T08-19-48"

# create CSV file with performance metrics
logger = PerformanceLogger(device)
logger.create_log(log_path, timestamp)
