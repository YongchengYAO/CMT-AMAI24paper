function mask_3d = cal_convertVoxCoor2Mask_3D(coor_vers_in, size_img, size_voxel)

% convert coordinates to volume
subs_vertices = round(coor_vers_in ./ size_voxel);
mask_3d = zeros(size_img);
idx_vertices = sub2ind(size_img, subs_vertices(:,1), subs_vertices(:,2), subs_vertices(:,3));
mask_3d(idx_vertices) = 1;

end