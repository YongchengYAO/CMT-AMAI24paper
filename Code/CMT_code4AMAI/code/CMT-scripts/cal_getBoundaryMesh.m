function FV = cal_getBoundaryMesh(img3d)
% ==============================================================================
% FUNCTION:
%     Surface reconstruction for a 3D ROI.
%
% INPUT:
%     - img3d: [uint8] a binary 3D image
%
% OUTPUT:
%     - FV: [structure]
%         - FV.faces: size=[nf, 3], the faces on the boundary
%         - FV.vertices: size=[nv, 3], the vertices on the boundary
%
% <<< Caution <<<
% (tested on Matlab 2021b (release 3))
% The build-in function "isosurface" returns (x,y) coordinates
% Other build-in functions in matlab may return (row, column) coordinates
% >>> Caution >>>
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

% get triangular mesh for the boundary of the ROI
FV = isosurface(~img3d, 0);

% change the orientation: the "isosurface" function have swap the first two dimensions
tmp = FV.vertices;
if ~isempty(tmp)
    FV.vertices(:,2) = tmp(:,1);
    FV.vertices(:,1) = tmp(:,2);

    % vertices positions should represent the subscripts of voxels
    subs = round(FV.vertices);
    faces = FV.faces;

    % remove duplicated vertices
    [subs, faces] = remove_duplicated_vertices(subs, faces); % MeshProcessingToolbox

    % remove unreferenced vertices
    [subs, faces] = remove_unreferenced_vertices(subs, faces); % MeshProcessingToolbox

    % save vertices and faces
    FV.faces = faces;
    FV.vertices = subs;
else
    FV.faces = [];
    FV.vertices = [];
end

end