#!/bin/bash

export codeFolder="/home/co-yong1/rds/hpc-work/Documents/CartiMorphToolbox/CMT_paper/code"
export prediction_dir="/home/co-yong1/rds/hpc-work/Documents/CartiMorphToolbox/Models_training/VoxelMorph/vxm_inference/Task002_csd3-OAIZIB-nnunet1000-vxm2000x3MSELNCC/temp2img_warpedTempSeg_Ts"
export GT_dir="/home/co-yong1/rds/hpc-work/Documents/CartiMorphToolbox/Models_training/VoxelMorph/vxm_data/Task002_csd3-OAIZIB-nnunet1000-vxm2000x3MSELNCC/labelsTs_mdcn"
export evalFolder="/home/co-yong1/rds/hpc-work/Documents/CartiMorphToolbox/CMT_paper/eval"
export warpingFieldFolder="/home/co-yong1/rds/hpc-work/Documents/CartiMorphToolbox/Models_training/VoxelMorph/vxm_inference/Task002_csd3-OAIZIB-nnunet1000-vxm2000x3MSELNCC/temp2img_warpingField_Ts"

# DSC
python $codeFolder/cal_DSC.py --prediction_dir $prediction_dir --GT_dir $GT_dir --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_DSC_summary.py --csv_path $evalFolder/eval_DSC.csv

# Jacobian determinant
python $codeFolder/cal_JacoDet.py --deformationField_dir $warpingFieldFolder --eval_dir $evalFolder 
python $codeFolder/cal_JacoDet_summary.py --csv_path $evalFolder/eval_JacoDet.csv 

# Surface Distance 
python $codeFolder/cal_surfaceDistance.py --prediction_dir $prediction_dir --GT_dir $GT_dir --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_surfaceDistance_summary.py --csv_path $evalFolder/eval_surfaceDistance.csv 

# Betti Error
python $codeFolder/cal_BettiNum.py --prediction_dir $prediction_dir --eval_dir $evalFolder --labels 1 2 3 4 5
python $codeFolder/cal_BettiNum_summary.py --csv_path $evalFolder/eval_Betti_numbers.csv --GT_Betti0 1 --GT_Betti1 0 


