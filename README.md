# pytorch-cb-loss

## Installation
python 3.9.12  
pytorch 1.11.0  
torchvision 0.12.0  
cudatoolkit 10.2.89  
tensorflow 2.9.1  
pillow  
numpy  
scikit-learn  
seaborn  
PyYAML  
matplotlib  

## Dataset
Check [this repository](https://github.com/richardaecn/class-balanced-loss#datasets) and download long-tailed CIFAR dataset.

## Train
Run this command.
```bash
sh train.sh
```
Change ```data_dir``` to your own path to long-tailed CIFAR data directory.   
Choose ```imbalance_factor``` from [0.1, 0.01, 0.02, 0.05, 0.005].

## Evaluation
Run this command.
```bash
sh test.sh
```
Change ```data_dir``` to your own path to long-tailed CIFAR data directory.   
Choose ```imbalance_factor``` from [0.1, 0.01, 0.02, 0.05, 0.005].  
Choose ```loss_type``` from ["softmax", "sigmoid", "focal"].
