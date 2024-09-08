function [subs_inter, subs_outer] = cal_initBoundarySplit_wBone(...
    mask_cartilage,...
    mask_bone)
% ==============================================================================
% FUNCTION:
%     Coarse separation of boundary.
%
% INPUT:
%     - mask_cartilage: [uint8] 3D cartilage mask
%     - mask_bone:  [uint8] 3D bone mask
%
% OUTPUT:
%     - subs_inter: [double] (nv, 3), the subscripts of voxels on the interface
%     - subs_outer: [double] (nv, 3), the subscripts of voxels on the outer surface
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
% ==============================================================================

% split boundary using sagittal slices (assuming RAS+ orientation)
[subs_inter_sag, subs_outer_sag] = cal_initBoundarySplit2D_wBone(mask_cartilage, mask_bone, 1);

% split boundary using coronal slices (assuming RAS+ orientation)
[subs_inter_cor, subs_outer_cor] = cal_initBoundarySplit2D_wBone(mask_cartilage, mask_bone, 2);

% combine results from sagittal and coronal slices
subs_inter = unique([subs_inter_sag; subs_inter_cor], 'rows');
subs_outer = unique([subs_outer_sag; subs_outer_cor], 'rows');
subs_outer(ismember(subs_outer, subs_inter, 'rows'), :) = [];
end