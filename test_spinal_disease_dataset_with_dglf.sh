#!/bin/bash
python -u test_ensemble_spark.py --data_dir=root_data_path/Spinal_disease_dataset --fold_ind=1 --model=DeepLabv3_plus_lil --beta 40.0 50.0 60.0 70.0 80.0 90.0 100.0 --task=multi-task --devices cuda:2 --ensemble_type=model-selection-mv --epochs=600 --batch_size=4 --use_weak_label --optimizer=Adam --learning_rate=5e-3

