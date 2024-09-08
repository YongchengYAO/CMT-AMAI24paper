#!/bin/bash

# testing
# you might need to change model path: --modelpath 

CUDA_VISIBLE_DEVICES=0 python ../Code-Vt/Test_LapIRN_diff.py --gpu 0 --dim 64 128 128 --savepath ../../../Results/LapIRN/diff/OAIZIB-temp2img --movedSegDir ../../../Results/LapIRN/diff/OAIZIB-temp2img-seg --flowDir ../../../Results/LapIRN/diff/OAIZIB-temp2img-flow --fixedDir ../../../Data/LapIRN/test --moving ../../../Data/LapIRN/template/template.nii.gz --movingSeg ../../../Data/LapIRN/template/templateSeg.nii.gz --modelpath ../Model/LapIRN_diff_fea7.pth
