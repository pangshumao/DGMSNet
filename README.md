# DGMSNet
This is the official implementation of the paper:

Shumao Pang, Chunlan Pang, Zhihai Su, Liyan Lin, Lei Zhao, Yangfan Chen, Yujia Zhou, Hai Lu, Qianjin Feng. [DGMSNet: Spine Segmentation for MR Image by a Detection-Guided Mixed-supervised Segmentation Network](https://www.sciencedirect.com/science/article/pii/S1361841521003066), Medical Image Analysis, 2022, 102261.

Website: https://www.researchgate.net/profile/Shumao_Pang2

## Requirements
- Pytorch 1.7.0
- Python 3.6.5
## Prepare dataset
Here, we will show how to prepare the Dataset-B (i.e., Spinal disease dataset).  
Step 1: Download the Spinal Disease Dataset from: <https://tianchi.aliyun.com/dataset/dataDetail?dataId=79463>.  
Step 2: Move and rename the downloaded dataset to ./data/Spinal_disease_dataset. The files structure is shown as follows:

data
> Spinal\_disease\_dataset
>> lumbar\_train150  
>> lumbar\_train51  
>> lumbar\_testA50  
>> lumbar\_testB50  
>> lumbar\_train150\_annotation.json  
>> lumbar\_train51\_annotation.json  
>> testA50\_sagittal\_image\_information.xlsx  
>> testB50\_sagittal\_image\_information.xlsx  
>> split\_ind\_fold1.npz  

>> in  
>>> mask  # Note that this is the manual segmentation annotation released by us for the lumbar\_testA50 and lumbar\_testB50.  
>>> keypoints

Step 3: Run the following commands in the terminal:  

```
cd DGMSNet  
python -u prepare_spinal_disease_dataset.py --data_dir=./data/Spinal_disease_dataset
```
## Training
run the following script in the terminal:

```
nohup ./train_spinal_disease_dataset.sh > train_spinal_disease_dataset.out &
```
## Test
To test the model without DGLF, please run the following script in the terminal:

```
nohup ./test_spinal_disease_dataset_without_dglf.sh > test_spinal_disease_dataset_without_dglf.out &
```

To test the model with DGLF, please run the following script in the terminal:

```
nohup ./test_spinal_disease_dataset_with_dglf.sh > test_spinal_disease_dataset_with_dglf.out &
```

## Acknowledgement
The core and nn_tools packages were modified from <https://github.com/wolaituodiban/spinal_detection_baseline.git> and <https://github.com/wolaituodiban/nn_tools.git> respectively. Thanks for these works.

## Citation
Shumao Pang, Chunlan Pang, Zhihai Su, Liyan Lin, Lei Zhao, Yangfan Chen, Yujia Zhou, Hai Lu, Qianjin Feng. [DGMSNet: Spine Segmentation for MR Image by a Detection-Guided Mixed-supervised Segmentation Network](https://www.sciencedirect.com/science/article/pii/S1361841521003066), Medical Image Analysis, 2022, 102261.
