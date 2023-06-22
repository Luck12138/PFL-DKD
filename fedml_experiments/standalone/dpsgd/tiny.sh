#!/bin/bash
python main_dpsgd.py --model 'resnet18' \
--dataset 'tiny' \
--partition_method 'dir' \
--partition_alpha 0.3 \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022 \
--cs 'random' \
--type "epoch"