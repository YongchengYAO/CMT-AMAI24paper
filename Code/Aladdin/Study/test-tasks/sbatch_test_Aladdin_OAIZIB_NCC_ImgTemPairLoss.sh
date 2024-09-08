#!/bin/bash

# task
export taskName="Aladdin_OAIZIB_NCC_ImgTemPairLoss"
export jobName="job1"


# folder
export dataFolder="../../../Data/Aladdin/OAIZIB-64x128x128"
export codeFolder="../../Code-Vt"
export modelFolder="../../Model/$taskName/$jobName"
export evalFolder="../../../../Results/Aladdin/Evaluation/$taskName/$jobName"
export warpingFieldFolder="$evalFolder/warpingField"
export warpedTempImgFolder="$evalFolder/movedTemp"
export warpedTempSegFolder="$evalFolder/movedTempSeg"
export targetImgSegFolder="$dataFolder/label_test"

if [ ! -d "$warpingFieldFolder" ]; then
	mkdir -p "$warpingFieldFolder"
fi
if [ ! -d "$warpedTempImgFolder" ]; then
	mkdir -p "$warpedTempImgFolder"
fi
if [ ! -d "$warpedTempSegFolder" ]; then
	mkdir -p "$warpedTempSegFolder"
fi


# GPU
export CUDA_VISIBLE_DEVICES=0
export gpu=0


# inference
python $codeFolder/Aladdin_OAIZIB_test.py --gpu $gpu --dataFolder $dataFolder --modelFolder $modelFolder --warpingFieldFolder $warpingFieldFolder --warpedTempImgFolder $warpedTempImgFolder --warpedTempSegFolder $warpedTempSegFolder


# evaluation
# DSC
python $codeFolder/cal_DSC.py --prediction_dir $warpedTempSegFolder --GT_dir $targetImgSegFolder --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_DSC_summary.py --csv_path $evalFolder/eval_DSC.csv 

# Jacobian determinant
python $codeFolder/cal_JacoDet.py --deformationField_dir $warpingFieldFolder --eval_dir $evalFolder 
python $codeFolder/cal_JacoDet_summary.py --csv_path $evalFolder/eval_JacoDet.csv 

# Surface Distance
python $codeFolder/cal_surfaceDistance.py --prediction_dir $warpedTempSegFolder --GT_dir $targetImgSegFolder --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_surfaceDistance_summary.py --csv_path $evalFolder/eval_surfaceDistance.csv 
