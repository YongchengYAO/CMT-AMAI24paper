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

clear;

% -------------------------------------
% configs
% -------------------------------------
model = 'Aladdin_OAIZIB_NCC_ImgTemPairLoss';
run = 'job1';

% dirs
dir_CMT_code = '../../CMT_code4AMAI/code';
dir_CMT_eval = '../../CMT_code4AMAI/eval';
dir_eval = fullfile('../../Evaluation', model, run);

% add functions
addpath(genpath(dir_CMT_code));

% data
dir_seg = fullfile(dir_eval, 'movedTempSeg');
caseIDs_dataset3 = [465,466,472,474,476,486,489,492,494,495,497,500]';


% ======
% label : ROI
% 1: Femur
% 2: FC
% 3: Tibia
% 4: mTC
% 5: lTC
% ======
cartilage_names = {'FemoralCartilage', 'mTibialCartilage', 'lTibialCartilage'};
cartilage_labels.FemoralCartilage = 2;
cartilage_labels.mTibialCartilage = 4;
cartilage_labels.lTibialCartilage = 5;

bone_names = {'Femur', 'Tibia'};
bone_labels.Femur = 1;
bone_labels.Tibia = 3;
% -------------------------------------


%% cal: eval_surfArea_warpedTempSeg
% list of cases
tmp = [dir(fullfile(dir_seg, "*.nii.gz")); dir(fullfile(dir_seg, "*.nii"))];
tmp = {tmp.name}';
list_cases = cell(length(caseIDs_dataset3),1);
% filter cases
count = 0;
for i = 1:length(tmp)
    currentID = sscanf(tmp{i}, 'oaizib_%d.nii.gz');
    if ismember(currentID, caseIDs_dataset3)
        count = count + 1;
        list_cases{count} = tmp{i};
    end
end
num_cases = length(list_cases);

num_cartilages = length(fieldnames(cartilage_labels));
surfArea_wTempSeg = zeros([num_cases, num_cartilages], 'double');
for i=1:num_cases
    fileName_wExt = list_cases{i};
    [~, tmp_fileName, tmp_ext] = fileparts(fileName_wExt);
    if strcmp(tmp_ext, ".gz")
        [~, fileName_woExt, ~] = fileparts(tmp_fileName);
        ext = ".nii.gz";
    else
        fileName_woExt = tmp_fileName;
        ext = ".nii";
    end

    % read NIfTI file
    niiInfo = niftiinfo(fullfile(dir_seg, fileName_wExt));
    seg = niftiread(niiInfo); % 3D array: segmentation

    % voxel size
    size_voxel = niiInfo.PixelDimensions;

    for j=1:num_cartilages
        % cartilage mask
        cartilageName = cartilage_names{j};
        cartilageLabel = cartilage_labels.(cartilageName);
        cartilageMask = zeros(size(seg), 'uint8');
        cartilageMask(seg==cartilageLabel) = 1;

        % bone mask
        boneMask = zeros(size(seg), 'uint8');
        switch cartilageName
            case 'FemoralCartilage'
                boneMask(seg==bone_labels.Femur) = 1;
            case 'mTibialCartilage'
                boneMask(seg==bone_labels.Tibia) = 1;
            case 'lTibialCartilage'
                boneMask(seg==bone_labels.Tibia) = 1;
        end

        % surface segmentation
        % Mesh_iC: interior surface (bone-cartilage interface)
        % Mesh_eC: exterior surface (outer cartilage surface)
        [Mesh_iC, Mesh_eC] = cal_surfaceSegmentation(cartilageMask, boneMask, size_voxel);

        % surface area
        surfArea_wTempSeg(i,j) = sum(cal_triMeshArea(Mesh_iC.vertices, Mesh_iC.faces));

    end
end

% save to csv file
surfArea_wTempSeg_table = array2table(surfArea_wTempSeg);
surfArea_wTempSeg_table.Properties.VariableNames = cartilage_names;
surfArea_wTempSeg_table.Properties.RowNames = list_cases;
csvFileName = fullfile(dir_eval, 'eval_surfArea_warpedTempSeg.csv');
writetable(surfArea_wTempSeg_table, csvFileName, 'WriteRowNames', true);



%% Compare the surface area of the warped template segmentation with that of the GT
csv_surfArea_GT = fullfile(dir_CMT_eval, 'eval_surfArea_GT.csv');
surfArea_GT_table = readtable(csv_surfArea_GT, 'ReadVariableNames', true, 'ReadRowNames', false);
surfArea_GT = surfArea_GT_table(:,2:end);

surfAreaRelDiff_table = (surfArea_wTempSeg - surfArea_GT) ./ surfArea_GT;
surfAreaRelDiff = table2array(surfAreaRelDiff_table);
means = mean(surfAreaRelDiff, 1);
meanRow_table = table(means(1), means(2), means(3), 'VariableNames', cartilage_names);
surfAreaRelDiff_table = [surfAreaRelDiff_table; meanRow_table];

% Save the table to a CSV file
surfAreaRelDiff_table.Properties.VariableNames = cartilage_names;
list_cases(end+1) = {'mean'};
surfAreaRelDiff_table.Properties.RowNames = list_cases;
csvFileName = fullfile(dir_eval, 'eval_surfAreaRelDiff.csv');
writetable(surfAreaRelDiff_table, csvFileName, 'WriteRowNames', true);
