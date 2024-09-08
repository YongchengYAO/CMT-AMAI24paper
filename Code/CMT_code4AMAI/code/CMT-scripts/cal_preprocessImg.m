function img_out = cal_preprocessImg(img_in, n_vox_tbr)
    % remove isolated voxels, defined as connected cluster of voxels with size less
    % than a threshold "n_vox_tbr"
    tmp = bwareaopen(img_in, n_vox_tbr, 26);  % conn=26
    img_out = uint8(imfill(tmp, 6, 'holes'));  % conn=6
end
