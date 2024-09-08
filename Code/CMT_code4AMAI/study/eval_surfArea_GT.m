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

clc;
clear;

% -------------------------------------
% configs
% -------------------------------------
% dirs
wd = '/Users/vincent/Documents/CartiMorphToolbox/CMT_paper';
dir_data = fullfile(wd, 'data');
dir_CMT_code = '/Users/vincent/Documents/CartiMorphToolbox/CMT_paper/code';
dir_eval = fullfile(wd, 'eval');

% add functions
addpath(genpath(dir_CMT_code));

% data 
dir_mesh_scB_GT = fullfile(dir_data, 'Task111_OAIZIB_scratch', 'Mesh_scB');

% ======
% label : ROI
% 1: Femur
% 2: FC
% 3: Tibia
% 4: mTC
% 5: lTC
% ======
cartilage_names = {'FemoralCartilage', 'mTibialCartilage', 'lTibialCartilage'};
num_cartilage = length(cartilage_names);
% -------------------------------------



% list of cases
tmp = dir(fullfile(dir_mesh_scB_GT, cartilage_names{1} ,"*.mat"));
tmp = {tmp.name}';
list_cases = cell(length(tmp),1);
for i = 1:length(tmp)
    list_cases{i,1} = tmp{i}(1:20);
end
num_cases = length(list_cases);

num_cartilages = length(cartilage_names);
Areas = zeros([num_cases, num_cartilages], 'double'); 
for i=1:num_cases
    for j=1:num_cartilage
        fileName = [list_cases{i} cartilage_names{j} '.mat'];
        file_mesh = fullfile(dir_mesh_scB_GT, cartilage_names{j}, fileName);
        Mesh_scB = load(file_mesh);
        % surface area
        Areas(i,j) = sum(cal_triMeshArea(Mesh_scB.vertices, Mesh_scB.faces));
    end
end

% save to csv file
Areas_table = array2table(Areas);
Areas_table.Properties.VariableNames = cartilage_names;
rowNames = cell(length(list_cases),1);
for k=1:length(tmp)
    rowNames{k,1} = [tmp{k}(1:10) '.nii.gz'];
end
Areas_table.Properties.RowNames = rowNames;

% Save the table to a CSV file
csvFileName = fullfile(dir_eval, 'eval_surfArea_GT.csv');
writetable(Areas_table, csvFileName, 'WriteRowNames', true);













