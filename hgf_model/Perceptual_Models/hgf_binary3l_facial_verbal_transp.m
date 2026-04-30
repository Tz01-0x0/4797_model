function [pvec, pstruct] = hgf_binary3l_facial_verbal_transp(r, ptrans)
% Transforms perceptual parameters from estimation space to native space.
% Adapted from hgf_binary3l_reward_social_transp.m: _r -> _f, _a -> _v
%
% --------------------------------------------------------------------------------------
% Original Copyright (C) 2012-2013 Christoph Mathys, Andreea Diaconescu TNU, UZH & ETHZ

pvec    = NaN(1,length(ptrans));
pstruct = struct;

% Facial stream parameters
pvec(1)        = ptrans(1);                            % mu2f_0
pstruct.mu2f_0 = pvec(1);
pvec(2)        = exp(ptrans(2));                       % sa2f_0
pstruct.sa2f_0 = pvec(2);
pvec(3)        = ptrans(3);                            % mu3f_0
pstruct.mu3f_0 = pvec(3);
pvec(4)        = exp(ptrans(4));                       % sa3f_0
pstruct.sa3f_0 = pvec(4);
pvec(5)        = tapas_sgm(ptrans(5),r.c_prc.kaub_f); % ka_f
pstruct.ka_f   = pvec(5);
pvec(6)        = ptrans(6);                            % om_f
pstruct.om_f   = pvec(6);
pvec(7)        = tapas_sgm(ptrans(7),r.c_prc.thub_f); % th_f
pstruct.th_f   = pvec(7);

% Verbal stream parameters
pvec(8)        = ptrans(8);                            % mu2v_0
pstruct.mu2v_0 = pvec(8);
pvec(9)        = exp(ptrans(9));                       % sa2v_0
pstruct.sa2v_0 = pvec(9);
pvec(10)       = ptrans(10);                           % mu3v_0
pstruct.mu3v_0 = pvec(10);
pvec(11)       = exp(ptrans(11));                      % sa3v_0
pstruct.sa3v_0 = pvec(11);
pvec(12)       = tapas_sgm(ptrans(12),r.c_prc.kaub_v); % ka_v
pstruct.ka_v   = pvec(12);
pvec(13)       = ptrans(13);                           % om_v
pstruct.om_v   = pvec(13);
pvec(14)       = tapas_sgm(ptrans(14),r.c_prc.thub_v); % th_v
pstruct.th_v   = pvec(14);

return;
