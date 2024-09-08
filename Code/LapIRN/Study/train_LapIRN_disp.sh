#!/bin/bash

# training
CUDA_VISIBLE_DEVICES=0 python ../Code-Vt/Train_LapIRN_disp.py --datapath ../../../Data/LapIRN/train --gpu 0 
