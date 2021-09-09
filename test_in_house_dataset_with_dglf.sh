#!/bin/bash
for fold_ind in 1 2 3 4 5;do
	for lr in 5e-3;do
		python -u test_ensemble.py --data_dir=root_data_path/In_house_dataset --fold_ind=${fold_ind} --model=DeepLabv3_plus_lil --beta 40.0 50.0 60.0 70.0 80.0 90.0 100.0 --task=multi-task --devices cuda:0 --ensemble_type=model-selection-mv --epochs=200 --batch_size=4 --use_weak_label --optimizer=Adam --learning_rate=${lr}
	done
done

