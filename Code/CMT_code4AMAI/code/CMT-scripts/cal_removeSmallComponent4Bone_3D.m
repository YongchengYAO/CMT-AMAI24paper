function mask = cal_removeSmallComponent4Bone_3D(mask)

CC_Mask = bwconncomp(mask, 26);
num_CC_Mask = length(CC_Mask.PixelIdxList);
if num_CC_Mask>1
    CC_size = zeros([num_CC_Mask, 1]);
    for i=1:num_CC_Mask
        CC_size(i, 1) = length(CC_Mask.PixelIdxList{i});
    end
    [~, idx_maxCC_Mask] = max(CC_size);
    tmp = CC_Mask.PixelIdxList;
    tmp(idx_maxCC_Mask) = [];
    CC_PixelIdxList_Mask_tbr = [];
    for i=1:length(tmp)
        CC_PixelIdxList_Mask_tbr = [CC_PixelIdxList_Mask_tbr; tmp{i}];
    end
    mask(CC_PixelIdxList_Mask_tbr) = 0;
end

end