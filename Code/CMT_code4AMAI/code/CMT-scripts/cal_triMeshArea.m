function facesArea = cal_triMeshArea(vertices, faces)
% ==============================================================================
% FUNCTION:
%     Calculate area of triangular mesh.
%
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

vector1 = vertices(faces(:, 2), :) - vertices(faces(:, 1), :);
vector2 = vertices(faces(:, 3), :) - vertices(faces(:, 1), :);
crossV1V2 = cross(vector1, vector2, 2);
facesArea = 0.5 * vecnorm(crossV1V2, 2, 2);
end