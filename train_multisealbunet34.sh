#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

for i in 0
do
    python train.py --device-ids 0 --limit 10000 --batch-size 4 --n-epochs 10 --fold $i --model MultiSEAlbuNet --multiple-output True
    python train.py --device-ids 0 --limit 10000 --batch-size 4 --n-epochs 15 --fold $i --lr 0.00001 --model MultiSEAlbuNet --multiple-output True
done
