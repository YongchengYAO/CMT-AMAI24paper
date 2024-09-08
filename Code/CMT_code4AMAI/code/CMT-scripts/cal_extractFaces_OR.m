function faces_out = cal_extractFaces_OR(faces_source, vers_source, vers_in)
% ==============================================================================
% FUNCTION:
%     Extract faces with at least one vertices in the target vertices set.
%
% INPUT:
%     - faces_source: (nf,3), faces of source mesh
%     - vers_source: (nv,3), vertices of source mesh
%     - vers_in: (nv_in, 3), target vertices
%
% OUTPUT:
%     - faces_out: (nf_out, 3), output faces whose entries are row-ID of vers_source
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

verIDs_source = transpose(1:size(vers_source,1));
verIDs = verIDs_source(ismember(vers_source, vers_in, 'rows'));
faces_out = faces_source(sum(ismember(faces_source, verIDs), 2)>=1, :);

end