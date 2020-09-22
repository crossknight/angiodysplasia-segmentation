#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python generate_masks.py --model_type MultiSEAlbuNet --model_path runs/debug/ --fold -1 --batch-size 1 --multiple-output True
python evaluate.py --target_path runs/debug/MultiSEAlbuNet/ > runs/debug/eval.log
