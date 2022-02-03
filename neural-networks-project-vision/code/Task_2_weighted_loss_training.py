# This script was used to execute the hyperparameter tuning experiments for R2U-Net
# with weighted loss to counteract class imbalance.

# only use specific GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import torch.optim as opt
import torch.nn as nn
from R2UNet import R2UNet
from ExperimentConfiguration import ExperimentConfiguration
from ExperimentLogger import ExperimentLogger
from ClassConfiguration import ClassConfiguration
from ExperimentRunner import ExperimentRunner
from ClassWeightCalculator import ClassWeightCalculator
from CityscapesLoader import CityscapesLoader

# specify device depending on availability of GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make the experiment deterministic
torch.manual_seed(3961459576686268241)
# save the seed to get identical dataset splits for weight calculation and training
seed = torch.random.initial_seed()

# compute the loss weights
trainSet, _, _ = CityscapesLoader().load()
weights = ClassWeightCalculator().calculate_weights(trainSet, 128).to(device)

# ensure the same dataset split for the experiment
torch.manual_seed(seed)

# setup the experiment
log_path = "../experiments/train_ids/task2_weighted_loss/"
optimizer_conf = ClassConfiguration(opt.Adam, {
    "lr": 1e-4
})
loss_conf = ClassConfiguration(nn.CrossEntropyLoss, {
    "ignore_index": 255, # ignore unlabeled pixels
    "weight": weights
})
model_conf = ClassConfiguration(R2UNet, {
    "channels": [3, 16, 32, 64, 128, 64, 32, 16, 19]
})
exp_conf = ExperimentConfiguration(model_conf, optimizer_conf, loss_conf,
                                   epochs=300, train_batch_size=32, val_batch_size=128, early_stop_patience=7)
logger = ExperimentLogger(log_path, exp_conf)

# run the experiment
runner = ExperimentRunner(logger, device)
runner.run()
