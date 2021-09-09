# DGMSNet
## This is the official implementation of DGMSNet: Spine Segmentation for MR Image by a Detection-Guided Mixed-supervised Segmentation Network, 2021.
## Requirements
- Pytorch
## Prepare dataset
Here, we will show how to prepare the Dataset-B (i.e., Spinal disease dataset).
1. Download the Spinal Disease Dataset from: https://tianchi.aliyun.com/dataset/dataDetail?dataId=79463.
2. Move and rename the downloaded dataset to ./data/Spinal_disease_dataset. The files structure is shown as follows:

> data
>> Spinal_disease_dataset
>>> lumbar_train150

>>> lumbar_train51

>>> lumbar_testA50

>>> lumbar_testB50

>>> lumbar_train150_annotation.json

>>> lumbar_train51_annotation.json

>>> testA50_sagittal_image_information.xlsx

>>> testB50_sagittal_image_information.xlsx

>>> split_ind_fold1.npz

>>> in
>>>> mask

3. Run the following commands in the terminal:

cd DGMSNet

python -u prepare_spinal_disease_dataset.py --data_dir=./data/Spinal_disease_dataset

## Training
run the following script in the terminal:

nohup ./train_spinal_disease_dataset.sh > train_spinal_disease_dataset.out &

## Test
To test the model without DGLF, please run the following script in the terminal:

nohup ./test_spinal_disease_dataset_without_dglf.sh > test_spinal_disease_dataset_without_dglf.out &

To test the model with DGLF, please run the following script in the terminal:

nohup ./test_spinal_disease_dataset_with_dglf.sh > test_spinal_disease_dataset_with_dglf.out &
