function pstruct = hgf_binary3l_facial_verbal_gamma_namep(pvec)
% Names perceptual parameters for the gamma model variant.

pstruct = struct;

pstruct.mu2f_0 = pvec(1);
pstruct.sa2f_0 = pvec(2);
pstruct.mu3f_0 = pvec(3);
pstruct.sa3f_0 = pvec(4);
pstruct.ka_f   = pvec(5);
pstruct.om_f   = pvec(6);
pstruct.th_f   = pvec(7);

pstruct.mu2v_0 = pvec(8);
pstruct.sa2v_0 = pvec(9);
pstruct.mu3v_0 = pvec(10);
pstruct.sa3v_0 = pvec(11);
pstruct.ka_v   = pvec(12);
pstruct.om_v   = pvec(13);
pstruct.th_v   = pvec(14);

pstruct.gamma  = pvec(15);

return;
