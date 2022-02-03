Marco Bjarne Schuster (7008806)(masc00008@stud.uni-saarland.de), Deepanshu Mehta (7011083)(deme00001@stud.uni-saarland.de)
# Guide to handed in files

| Path in ZIP file | Content |
| ---------------- | ------- |
| `/nn-report.pdf` | Scientific report about task 2 and 3 |
| `/code` | Completed Jupyter notebook for task 1, final model weights for task 1, Python code for task 2 and 3 |
| `/code/model_state_epoch_20.pt` | Final model weights for task 1 |
| `/code/Task_2_basic_training.py` | Main script used to train and tune R2U-Net |
| `/code/Task_3_basic_training.py` | Main script used to train and tune BCDU-Net |
| `/code/Task_2_weighted_loss_training.py` | Main script used to train R2U-Net with weighted cross entropy loss |
| `/code/Task_2_3_evaluation.py` | Main script used to evaluate R2U-Net and BCDU-Net on the test set |
| `/experiments/train_ids` | experiment configurations, final models and evaluation metrics for task 2 and 3 |
| `/experiments/train_ids/*/*-log.json` | experiment configurations and training logs |
| `/experiments/train_ids/*/*-model.pt` | final models |
| `/experiments/train_ids/*/*-eval.csv` | evaluation metrics |
| `/experiments/train_ids/task2_basic_training` | files for best R2U-Net (experiment 1) |
| `/experiments/train_ids/task3_basic_training` | files for best BCDU-Net (experiment 7) |
| `/experiments/train_ids/task2_weighted_loss` | files for re-executed experiment 1 with weighted cross entropy loss |
