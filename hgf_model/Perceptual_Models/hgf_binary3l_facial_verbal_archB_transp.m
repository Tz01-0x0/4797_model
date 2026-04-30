function [pvec, pstruct] = hgf_binary3l_facial_verbal_archB_transp(r, ptrans)
% Transform parameters from estimation to native space for ArchB.
% Parameter vector shape is identical to ArchA — delegate to ArchA transp.
[pvec, pstruct] = hgf_binary3l_facial_verbal_transp(r, ptrans);
return;
