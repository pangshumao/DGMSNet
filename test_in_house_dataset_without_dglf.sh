#!/bin/bash
for fold_ind in 1 2 3 4 5;do
	for lr in 5e-3;do
		for beta in 50.0;do
			python -u test.py --data_dir=root_data_path/In_house_dataset --fold_ind=${fold_ind} --model=DeepLabv3_plus_lil --beta=${beta} --task=multi-task --devices cuda:1 --epochs=200 --batch_size=4 --use_weak_label --optimizer=Adam --learning_rate=${lr}
		done
	done
done

