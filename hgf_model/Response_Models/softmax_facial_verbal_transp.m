function [pvec, pstruct] = softmax_facial_verbal_transp(r, ptrans)
% Transforms observation parameters from estimation space to native space.
% Identical to softmax_reward_social_transp.m
%
% -------------------------------------------------------------------------
% Original Copyright (C) 2012 Christoph Mathys, TNU, UZH & ETHZ

pvec    = NaN(1,length(ptrans));
pstruct = struct;

pvec(1)      = exp(ptrans(1));    % ze1 (zeta)
pstruct.ze1  = pvec(1);
pvec(2)      = exp(ptrans(2));    % beta
pstruct.beta = pvec(2);

return;
