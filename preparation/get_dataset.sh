#!/bin/bash

# From `https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/get_datasets.sh`

####################################
#   GET PTBXL DATABASE
####################################
mkdir -p ../data/PTBXL/raw
cd ../data/PTBXL/raw
wget https://storage.googleapis.com/ptb-xl-1.0.1.physionet.org/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ptbxl
cd ../../../preparation

####################################
#   GET CPSC2018 DATABASE
####################################

mkdir -p ../data/CPSC2018/raw
cd ../data/CPSC2018/raw
wget http://2018.icbeb.org/file/REFERENCE.csv
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip
unzip TrainingSet1.zip
unzip TrainingSet2.zip
unzip TrainingSet3.zip
cd ../../../preparation
