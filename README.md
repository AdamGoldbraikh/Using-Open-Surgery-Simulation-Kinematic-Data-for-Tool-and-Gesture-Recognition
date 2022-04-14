# Using-Open-Surgery-Simulation-Kinematic-Data-for-Tool-and-Gesture-Recognition (IJCARS)
A PyTorch implementation of the paper [Using-Open-Surgery-Simulation-Kinematic-Data-for-Tool-and-Gesture-Recognition](https://link.springer.com/article/10.1007/s11548-022-02615-1).

![Sensors' Locations](figures/sensor_localization_v2.png)

## Install
This implementation uses Python 3.6 and the following packages:
```
opencv-python==4.2.0.32
optuna==2.8.0
numpy==1.19.5
torch==1.8.1
pandas==1.1.5
wandb==0.10.33
tqdm==4.61.2
termcolor==1.1.0
```
We recommend to use conda to deploy the environment

## Dataset
Will be published soon...

## Run the code
To train and test the model on all the splits run:
```
python train_experiment.py
```
The visualization result is located in `summaries/APAS/experiment_name`,
Where `experiment_name` is a string describing the experiment: the network type, whether it is online, etc.
