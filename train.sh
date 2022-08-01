#!/bin/sh

data_dir="/work/ohashi/LongTail_CIFAR"
dataset_type="10"
imbalance_factor="0.005"

loss_type="softmax"
for beta in 0 0.9 0.99 0.999 0.9999
do
    echo "python train.py -data_dir ${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor} -save_path logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type} -beta ${beta} -loss_type ${loss_type}"
    python train.py -data_dir "${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor}" -save_path "logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}" -beta "${beta}" -loss_type "${loss_type}"
done

loss_type="sigmoid"
for beta in 0 0.9 0.99 0.999 0.9999
do
    echo "python train.py -data_dir ${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor} -save_path logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type} -beta ${beta} -loss_type ${loss_type}"
    python train.py -data_dir "${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor}" -save_path "logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}" -beta "${beta}" -loss_type "${loss_type}"
done

loss_type="focal"
for beta in 0 0.9 0.99 0.999 0.9999
do
    for gamma in 0.5 1.0 2.0 
    do
        echo "python train.py -data_dir ${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor} -save_path logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}${gamma} -beta ${beta} -gamma ${gamma}" -loss_type "${loss_type}"
        python train.py -data_dir "${data_dir}/cifar-${dataset_type}-data-im-${imbalance_factor}" -save_path "logs/cifar${dataset_type}-im-${imbalance_factor}-${loss_type}${gamma}" -beta "${beta}" -gamma "${gamma}" -loss_type "${loss_type}"
    done
done