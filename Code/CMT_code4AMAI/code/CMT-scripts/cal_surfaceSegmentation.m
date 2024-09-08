function [FV_inner, FV_outer] = cal_surfaceSegmentation(...
    mask_cartilage,...
    mask_bone,...
    size_voxel)
% ------------------------------------------------------------------------------
% Matlab Version: 2019b or later
%
% Last updated on: 26-Jul-2024
% Based on CMT code
%
% Author:
% Yongcheng YAO (yc.yao@ed.ac.uk)
%
% Copyright 2020 Yongcheng YAO
% ------------------------------------------------------------------------------


%% Preprocessing of segmentation masks
% image preprocessing: isolated voxels removal and filling holes within the mask
% set threshold for isolated voxels removal: regions with area less than 1 mm3
threshold_isoVox = ceil(1 / (size_voxel(1)*size_voxel(2)*size_voxel(3)));
mask_cartilage = cal_preprocessImg(mask_cartilage, threshold_isoVox);
mask_bone = cal_preprocessImg(mask_bone, threshold_isoVox);


%% Split the cartilage boundary
% --------------------------------------------------------------
% iC: interior surface of cartilage / bone-cartilage interface
% eC: exterior surface of cartilage
% C: cartilage surface
% scB: subchondral bone surface, the bone surface covered by cartilage
% FCL: reconstructed full-thickness cartilage loss surface
% B: bone surface
% --------------------------------------------------------------
[FV_inner, FV_outer] = cal_splitBoundary3D_wBone(mask_cartilage, mask_bone, size_voxel);

end