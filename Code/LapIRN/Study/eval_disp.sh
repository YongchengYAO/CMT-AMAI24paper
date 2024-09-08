#!/bin/bash

export variant='disp'

export codeFolder="../Code-Vt"
export prediction_dir="../../../Results/LapIRN/$variant/OAIZIB-temp2img-seg"
export GT_dir="../../../Data/LapIRN/test-label"
export evalFolder="../../../Results/LapIRN/$variant/eval"
export warpingFieldFolder="../../../Results/LapIRN/$variant/OAIZIB-temp2img-flow"

# DSC
python $codeFolder/cal_DSC.py --prediction_dir $prediction_dir --GT_dir $GT_dir --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_DSC_summary.py --csv_path $evalFolder/eval_DSC.csv

# Jacobian determinant
python $codeFolder/cal_JacoDet.py --deformationField_dir $warpingFieldFolder --eval_dir $evalFolder 
python $codeFolder/cal_JacoDet_summary.py --csv_path $evalFolder/eval_JacoDet.csv 

# Surface Distance
python $codeFolder/cal_surfaceDistance.py --prediction_dir $prediction_dir --GT_dir $GT_dir --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_surfaceDistance_summary.py --csv_path $evalFolder/eval_surfaceDistance.csv
