function [pvec, pstruct] = hgf_binary3l_facial_verbal_archC_transp(r, ptrans)
% Transform parameters from estimation to native space for ArchC.
% Parameter vector shape is identical to ArchA — delegate.
[pvec, pstruct] = hgf_binary3l_facial_verbal_transp(r, ptrans);
return;
