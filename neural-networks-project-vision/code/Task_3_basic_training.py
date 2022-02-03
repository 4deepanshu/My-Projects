# This script was used to execute the hyperparameter tuning experiments for BCDU-Net.

# only use specific GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import torch.optim as opt
import torch.nn as nn
from BCDUNet import BCDUNet
from ExperimentConfiguration import ExperimentConfiguration
from ExperimentLogger import ExperimentLogger
from ClassConfiguration import ClassConfiguration
from ExperimentRunner import ExperimentRunner

# specify device depending on availability of GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make the experiment deterministic
torch.manual_seed(3961459576686268241)

# setup the experiment
log_path = "../experiments/train_ids/task3_more_channels/"
optimizer_conf = ClassConfiguration(opt.Adam, {
})
loss_conf = ClassConfiguration(nn.CrossEntropyLoss, {
    "ignore_index": 255, # ignore unlabeled pixels
})
model_conf = ClassConfiguration(BCDUNet, {
    "channels": [3, 32, 64, 128, 256, 128, 64, 32, 19]
})
exp_conf = ExperimentConfiguration(model_conf, optimizer_conf, loss_conf,
                                   epochs=300, train_batch_size=12, val_batch_size=32, early_stop_patience=7)
logger = ExperimentLogger(log_path, exp_conf)

# run the experiment
runner = ExperimentRunner(logger, device)
runner.run()
