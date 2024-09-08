function [marginEdges, marginFaces] = cal_detectSurfaceBoundary(faces_in)
% ==============================================================================
% FUNCTION:
%
% INPUT:
%     - faces_in: (nf_in, 3), the input faces
%
% OUTPUT:
%     - marginEdges: size=[ne, 2], the edge of the surface boundary
%     - marginFaces: size=[nf, 3], the marginal faces
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


% (the following two matrixs are corresponding in rows)
% ----------------------------------------------------
% get edges from faces
edges = cal_Faces2Edges(faces_in);
% find neighboring faces for each edge
[neighFaces_cell, n_neighFaces] = cal_neighFaces4Edges(faces_in);
% ----------------------------------------------------

% find edges with only one adjacent face
idx_marginEdges = n_neighFaces==1;
marginEdges = edges(idx_marginEdges, :);

% find faces on the edge
idx_marginFaces = cell2mat(neighFaces_cell(idx_marginEdges));
marginFaces = faces_in(idx_marginFaces, :);

end