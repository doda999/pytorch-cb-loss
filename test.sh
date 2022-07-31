#!/bin/sh

data_dir="/work/ohashi/LongTail_CIFAR"
dataset_type="10"
imbalance_factor="0.005"
loss_type="softmax"
beta="0.9"
gamma="1.0"

if [ ${loss_type} != "focal" ]; then
    CUDA_VISIBLE_DEVICES=8 python test.py -data_dir "${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor}" -save_path "logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}" -checkpoint "logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}/model_final_${beta}.pth" -beta "${beta}"
else
    CUDA_VISIBLE_DEVICES=8 python test.py -data_dir "${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor}" -save_path "logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}${gamma}" -checkpoint "logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}${gamma}/model_final_${beta}.pth" -beta "${beta}"
fi