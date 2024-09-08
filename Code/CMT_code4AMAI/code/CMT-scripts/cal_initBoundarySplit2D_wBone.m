function [subscripts_inter, subscripts_outer] = cal_initBoundarySplit2D_wBone(...
    mask_cartilage,...
    mask_bone,...
    idx_slicingDim)

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

% get the image size of segmentation mask
size_img = size(mask_cartilage);

% grow the bone ROI so that there is no gap between bone and cartilage ROIs
mask_bone_dilated = imdilate(mask_bone, strel('sphere', 1));
mask_bone_dilated = mask_bone_dilated & ~mask_cartilage; % logical
mask_bone_dilated = bwareaopen(mask_bone_dilated, round(sum(mask_bone_dilated(:))/10), 26); % logical
mask_cartilage_bone = mask_cartilage | mask_bone_dilated; % logical
mask_cartilage_bone = imfill(mask_cartilage_bone, 'holes'); % logical
mask_bone_grown = uint8(mask_cartilage_bone) - uint8(mask_cartilage); % uint8
mask_nonBone = 1 - mask_bone_grown;  % uint8

% get the boundary of cartilage ROI
boundary_cartilage = cal_getBoundary2D(mask_cartilage, idx_slicingDim); % uint8

% get the bone-cartilage interface
boundary_nonBone =  cal_getBoundary2D(mask_nonBone, idx_slicingDim); % uint8
boundary_inter = uint8(boundary_cartilage & boundary_nonBone); % uint8

% get the subscripts of voxels on the bone-cartilage interface
idx_inter = find(boundary_inter);
[sub1_inter, sub2_inter, sub3_inter] = ind2sub(size_img, idx_inter);
subscripts_inter = cat(2, sub1_inter, sub2_inter, sub3_inter);

% get the outer boundary
boundary_outer = boundary_cartilage - boundary_inter;  % uint8

% get the subscripts of voxels on the outer boundary
idx_outer = find(boundary_outer);
[sub1_outer, sub2_outer, sub3_outer] = ind2sub(size_img, idx_outer);
subscripts_outer = cat(2, sub1_outer, sub2_outer, sub3_outer);
end