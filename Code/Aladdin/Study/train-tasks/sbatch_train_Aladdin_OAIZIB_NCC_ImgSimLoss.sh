#!/bin/bash

# task
export taskName="Aladdin_OAIZIB_NCC_ImgSimLoss"
export jobName="job1"


# folder
export dataFolder="../../../Data/Aladdin/OAIZIB-64x128x128"
export codeFolder="../../Code-Vt"
export modelFolder="../../Model/$taskName/$jobName"
export TBFolder="../../Tensorboard"

if [ ! -d "$modelFolder" ] 
then 
    mkdir -p "$modelFolder"
fi

if [ ! -d "$TBFolder" ]
then
    mkdir -p "$TBFolder"
fi

if [ ! -d "../../Logs" ]
then
    mkdir -p "../../Logs"
fi


# model
export epochs=1000
export batchSize=2
export lr=1e-4
export regFactor=20000
export simFactor=10
export templatePairSimFactor=0
export imagePairSimFactor=0
export simLossType='NCC'
export savePerEpoch=100


# GPU
export CUDA_VISIBLE_DEVICES=0,1
export gpu=1


# training
CUDA_VISIBLE_DEVICES=0,1 python $codeFolder/Aladdin_OAIZIB_train.py --taskName "$taskName" --jobName "$jobName" --TBFolder "$TBFolder" --gpu "$gpu" --dataFolder "$dataFolder" --modelFolder "$modelFolder" --epochs "$epochs" --batch-size "$batchSize" --lr "$lr" --reg-factor "$regFactor" --sim-factor "$simFactor" --template-pair-sim-factor "$templatePairSimFactor" --image-pair-sim-factor "$imagePairSimFactor" --sim-loss "$simLossType" --save-per-epoch "$savePerEpoch"
