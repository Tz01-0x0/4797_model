function [pvec, pstruct] = hgf_binary3l_facial_verbal_gamma_transp(r, ptrans)
% Transforms perceptual parameters from estimation space to native space.
% Extends hgf_binary3l_facial_verbal_transp.m with gamma (param 15).

pvec    = NaN(1,length(ptrans));
pstruct = struct;

% Facial stream
pvec(1)        = ptrans(1);                            pstruct.mu2f_0 = pvec(1);
pvec(2)        = exp(ptrans(2));                       pstruct.sa2f_0 = pvec(2);
pvec(3)        = ptrans(3);                            pstruct.mu3f_0 = pvec(3);
pvec(4)        = exp(ptrans(4));                       pstruct.sa3f_0 = pvec(4);
pvec(5)        = tapas_sgm(ptrans(5),r.c_prc.kaub_f); pstruct.ka_f   = pvec(5);
pvec(6)        = ptrans(6);                            pstruct.om_f   = pvec(6);
pvec(7)        = tapas_sgm(ptrans(7),r.c_prc.thub_f); pstruct.th_f   = pvec(7);

% Verbal stream
pvec(8)        = ptrans(8);                            pstruct.mu2v_0 = pvec(8);
pvec(9)        = exp(ptrans(9));                       pstruct.sa2v_0 = pvec(9);
pvec(10)       = ptrans(10);                           pstruct.mu3v_0 = pvec(10);
pvec(11)       = exp(ptrans(11));                      pstruct.sa3v_0 = pvec(11);
pvec(12)       = tapas_sgm(ptrans(12),r.c_prc.kaub_v); pstruct.ka_v  = pvec(12);
pvec(13)       = ptrans(13);                           pstruct.om_v   = pvec(13);
pvec(14)       = tapas_sgm(ptrans(14),r.c_prc.thub_v); pstruct.th_v  = pvec(14);

% Gamma — logit transform with upper bound
pvec(15)       = tapas_sgm(ptrans(15),r.c_prc.gammaub); pstruct.gamma = pvec(15);

return;
