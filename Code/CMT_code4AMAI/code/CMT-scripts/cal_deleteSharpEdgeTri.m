function faces_out = cal_deleteSharpEdgeTri(faces_in)
% ==============================================================================
% FUNCTION:
%     Remove faces whose two or more edges are marginal edges of the mesh
%
% INPUT:
%     - faces_in: (nf_in, 3), the input faces
%
% OUTPUT:
%     - faces_out: (nf_out, 3), the ouput faces
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

% remove all "sharp triangles" on the edge
flag_continue = true;
faces_out_tmp = faces_in;
while flag_continue
    % detect surface edge
    [marginEdges, ~] = cal_detectSurfaceBoundary(faces_out_tmp);

    % find faces to be deleted
    idx_versOnMarginEdges = marginEdges(:);
    tmp = ismember(faces_out_tmp, idx_versOnMarginEdges);
    tmp = tmp(:,1) & tmp(:,2) & tmp(:,3);
    faces_tbd = faces_out_tmp(tmp, :);

    if ~isempty(faces_tbd)
        % remove faces whose two or more edges are marginal edges of the mesh
        faces_out_tmp = remove_triangles(faces_tbd, faces_out_tmp, 'explicit'); % MeshProcessingToolbox
    else
        flag_continue = false;
    end
end

faces_out = faces_out_tmp;

end