#!/bin/bash

# -------------------
# set variables
export dir_CartiMorphToolbox="path/to/working/directory/of/CartiMorphToolbox"
export TaskName="Task002_csd3-OAIZIB-nnunet1000-vxm2000x3MSELNCC"
# -------------------

export codeFolder="../code"
export prediction_dir="$dir_CartiMorphToolbox/Models_training/VoxelMorph/vxm_inference/$TaskName/temp2img_warpedTempSeg_Ts"
export GT_dir="$dir_CartiMorphToolbox/Models_training/VoxelMorph/vxm_data/$TaskName/labelsTs_mdcn"
export evalFolder="../../../Results/CMT/eval"
export warpingFieldFolder="$dir_CartiMorphToolbox/Models_training/VoxelMorph/vxm_inference/$TaskName/temp2img_warpingField_Ts"

# DSC
python $codeFolder/cal_DSC.py --prediction_dir $prediction_dir --GT_dir $GT_dir --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_DSC_summary.py --csv_path $evalFolder/eval_DSC.csv

# Jacobian determinant
python $codeFolder/cal_JacoDet.py --deformationField_dir $warpingFieldFolder --eval_dir $evalFolder 
python $codeFolder/cal_JacoDet_summary.py --csv_path $evalFolder/eval_JacoDet.csv 

# Surface Distance 
python $codeFolder/cal_surfaceDistance.py --prediction_dir $prediction_dir --GT_dir $GT_dir --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_surfaceDistance_summary.py --csv_path $evalFolder/eval_surfaceDistance.csv 
