#!/bin/bash
for lr in 5e-3;do
	for beta in 40.0 50.0 60.0 70.0 80.0 90.0 100.0;do
		python -u train.py --data_dir=./data/Spinal_disease_dataset --fold_ind=1 --model=DeepLabv3_plus_lil --beta=${beta} --task=multi-task --devices cuda:1 --epochs=600 --batch_size=4 --use_weak_label --optimizer=Adam --learning_rate=${lr}
	done
done
i
