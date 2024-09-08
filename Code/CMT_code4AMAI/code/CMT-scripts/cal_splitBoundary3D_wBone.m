function [FV_inner, FV_outer]...
    = cal_splitBoundary3D_wBone(...
    mask_cartilage,...
    mask_bone,...
    size_voxel)
% ==============================================================================
% FUNCTION:
%     1. Split the cartilage boundary into two parts:
%         (1) the bone-cartilage interface
%         (2) the outer boundary
%     2. Triangulation for two boundaries
%
% INPUT:
%     - mask_cartilage: [uint8] 3D cartilage mask
%     - mask_bone:  [uint8] 3D bone mask
%     - size_voxel: [double] the spatial resolution of a voxel
%
% OUTPUT:
%        (the following structure variables have same fields "faces" and "vertices")
%     - FV_inner: [structure]
%     - FV_outer: [structure]
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


%% Initial separation of cartilage boundary
%------------------------------------------------------------------------------------------
% coarse surface segmentation
[subs_inner_coarse, subs_outer_coarse] = cal_initBoundarySplit_wBone(...
    mask_cartilage,...
    mask_bone);

% convert subscripts to coordinates
vers_inner_coarse = subs_inner_coarse .* size_voxel;
vers_outer_coarse  = subs_outer_coarse .* size_voxel;
%------------------------------------------------------------------------------------------



%% Cartilage surface reconstruction
%------------------------------------------------------------------------------------------
% get boundary mesh
FV_cartilage = cal_getBoundaryMesh(mask_cartilage);

% convert subscripts to coordinates
subs_cartilage = FV_cartilage.vertices;
vers_cartilage = subs_cartilage .* size_voxel;
FV_cartilage.vertices = vers_cartilage;

% faces
faces_cartilage = FV_cartilage.faces;
%------------------------------------------------------------------------------------------



%%  Bone-cartilage interface reconstruction
%------------------------------------------------------------------------------------------
% coarse interface mesh
faces_inner_coarse = cal_extractFaces_AND(faces_cartilage, vers_cartilage, vers_inner_coarse);

% surface close on the coarse interface mesh
iteration_dilation = 4;
iteration_erosion = 4;
faces_inner = cal_surfaceClosing(...
    faces_inner_coarse,...
    faces_cartilage,...
    iteration_dilation,...
    iteration_erosion);

% (!!!important!!!) ---
% (the removal of duplicated faces must be after the surface reconstruction)
faces_inner = remove_duplicated_triangles(faces_inner);  % MeshProcessingToolbox
% (!!!important!!!) ---

% remove sharp-edge faces
for iter = 1:5
    faces_inner = cal_deleteSharpEdgeTri(faces_inner);
end

% add faces whose two edges are the edges of the interface mesh
faces_inner = cal_extractFaces_AND(faces_cartilage, vers_cartilage, vers_cartilage(unique(faces_inner(:)), :));

% update vertices of interface mesh (MeshProcessingToolbox)
[vers_inner, faces_inner] = remove_unreferenced_vertices(vers_cartilage, faces_inner);

% the mesh
FV_inner.faces = faces_inner;
FV_inner.vertices = vers_inner;
%------------------------------------------------------------------------------------------



%%  Outer cartilage surface reconstruction
%------------------------------------------------------------------------------------------
% coarse outer cartilage surface
faces_outer_coarse = cal_extractFaces_AND(faces_cartilage, vers_cartilage, vers_outer_coarse);

% restricted surface dilation
faces_outer = cal_surfaceDilation_restricted(...
    faces_outer_coarse,...
    faces_cartilage,...
    vers_cartilage,...
    vers_inner);

% add more faces to the outer cartilage surface
faces_outer = cal_extractFaces_OR(faces_cartilage, vers_cartilage, vers_cartilage(unique(faces_outer(:)), :));

% (!!!important!!!) ---
% (the removal of duplicated faces must be after the surface reconstruction)
faces_outer = remove_duplicated_triangles(faces_outer);  % MeshProcessingToolbox
% (!!!important!!!) ---

% update vertices of outer cartilage surface (MeshProcessingToolbox)
[vers_outer, faces_outer] = remove_unreferenced_vertices(vers_cartilage, faces_outer);

% the mesh
FV_outer.faces = faces_outer;
FV_outer.vertices = vers_outer;
%------------------------------------------------------------------------------------------

end