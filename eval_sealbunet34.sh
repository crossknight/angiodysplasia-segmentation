#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python generate_masks.py --model_type SEAlbuNet --model_path runs/debug/ --fold -1 --batch-size 1
python evaluate.py --target_path runs/debug/SEAlbuNet/ > runs/debug/eval.log
