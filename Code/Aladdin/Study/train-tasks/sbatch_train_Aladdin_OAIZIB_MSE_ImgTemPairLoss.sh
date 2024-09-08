#!/bin/bash

# task
export taskName="Aladdin_OAIZIB_MSE_ImgTemPairLoss"
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
export templatePairSimFactor=5
export imagePairSimFactor=5
export simLossType='SSD'
export savePerEpoch=100


# GPU
export CUDA_VISIBLE_DEVICES=0
export gpu=0


# training
python $codeFolder/Aladdin_OAIZIB_train.py --taskName "$taskName" --jobName "$jobName" --TBFolder "$TBFolder" --gpu "$gpu" --dataFolder "$dataFolder" --modelFolder "$modelFolder" --epochs "$epochs" --batch-size "$batchSize" --lr "$lr" --reg-factor "$regFactor" --sim-factor "$simFactor" --template-pair-sim-factor "$templatePairSimFactor" --image-pair-sim-factor "$imagePairSimFactor" --sim-loss "$simLossType" --save-per-epoch "$savePerEpoch"
