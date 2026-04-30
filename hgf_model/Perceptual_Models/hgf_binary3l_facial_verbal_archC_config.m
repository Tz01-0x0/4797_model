function c = hgf_binary3l_facial_verbal_archC_config
% Configuration for Architecture C (Hybrid) 3-level binary HGF.
% Shared x3_f across three equivalence groups, separate x2_f per group.
% kappa_f, omega_f, theta_f are shared. Verbal HGF unchanged from ArchA.
% Parameter vector shape is IDENTICAL to ArchA (14 params).

c = struct;
c.model = 'hgf_binary3l_facial_verbal_archC';

c.kaub_f = 1;
c.thub_f = 1;
c.kaub_v = 1;
c.thub_v = 1;

c.mu2f_0mu = 0;   c.mu2f_0sa = 0;
c.mu2v_0mu = 0;   c.mu2v_0sa = 0;

c.logsa2f_0mu = log(1);   c.logsa2f_0sa = 0;
c.logsa2v_0mu = log(1);   c.logsa2v_0sa = 0;

c.mu3f_0mu = 1;   c.mu3f_0sa = 0;
c.mu3v_0mu = 1;   c.mu3v_0sa = 0;

c.logsa3f_0mu = log(1);   c.logsa3f_0sa = 0;
c.logsa3v_0mu = log(1);   c.logsa3v_0sa = 0;

c.logitkamu_f = 0;   c.logitkasa_f = 1;
c.logitkamu_v = 0;   c.logitkasa_v = 1;

c.ommu_f = -4;   c.omsa_f = 0;
c.ommu_v = -4;   c.omsa_v = 0;

c.logitthmu_f = 0.25;   c.logitthsa_f = 1;
c.logitthmu_v = 0.25;   c.logitthsa_v = 1;

c.priormus = [
    c.mu2f_0mu, c.logsa2f_0mu, c.mu3f_0mu, c.logsa3f_0mu, ...
    c.logitkamu_f, c.ommu_f, c.logitthmu_f, ...
    c.mu2v_0mu, c.logsa2v_0mu, c.mu3v_0mu, c.logsa3v_0mu, ...
    c.logitkamu_v, c.ommu_v, c.logitthmu_v];

c.priorsas = [
    c.mu2f_0sa, c.logsa2f_0sa, c.mu3f_0sa, c.logsa3f_0sa, ...
    c.logitkasa_f, c.omsa_f, c.logitthsa_f, ...
    c.mu2v_0sa, c.logsa2v_0sa, c.mu3v_0sa, c.logsa3v_0sa, ...
    c.logitkasa_v, c.omsa_v, c.logitthsa_v];

c.prc_fun        = @hgf_binary3l_facial_verbal_archC;
c.transp_prc_fun = @hgf_binary3l_facial_verbal_archC_transp;

return;
