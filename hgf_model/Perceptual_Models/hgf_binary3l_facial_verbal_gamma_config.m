function c = hgf_binary3l_facial_verbal_gamma_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configuration for the 3-level binary HGF with gamma (generalization) parameter.
% Gamma is FREE — estimated from data.
%
% Free parameters: kappa_f, theta_f, kappa_v, theta_v, gamma (5 perceptual params)
% Fixed: omega_f = omega_v = -4, all initial conditions
%
% Model comparison variants (create separate config files or fix gamma):
%   gamma free (this file)  — M4: partial generalization
%   gamma = 1 (fix)         — M3: full generalization (unified)
%   gamma = 0 (fix)         — M2: no generalization (separated)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = struct;
c.model = 'hgf_binary3l_facial_verbal_gamma';

% Upper bounds
c.kaub_f = 1;
c.thub_f = 1;
c.kaub_v = 1;
c.thub_v = 1;
c.gammaub = 1;   % gamma bounded between 0 and 1

% Facial stream priors
c.mu2f_0mu = 0;        c.mu2f_0sa = 0;
c.logsa2f_0mu = log(1); c.logsa2f_0sa = 0;
c.mu3f_0mu = 1;         c.mu3f_0sa = 0;
c.logsa3f_0mu = log(1); c.logsa3f_0sa = 0;
c.logitkamu_f = 0;      c.logitkasa_f = 1;   % kappa FREE
c.ommu_f = -4;          c.omsa_f = 0;         % omega FIXED
c.logitthmu_f = 0.25;   c.logitthsa_f = 1;   % theta FREE

% Verbal stream priors
c.mu2v_0mu = 0;         c.mu2v_0sa = 0;
c.logsa2v_0mu = log(1); c.logsa2v_0sa = 0;
c.mu3v_0mu = 1;         c.mu3v_0sa = 0;
c.logsa3v_0mu = log(1); c.logsa3v_0sa = 0;
c.logitkamu_v = 0;      c.logitkasa_v = 1;   % kappa FREE
c.ommu_v = -4;          c.omsa_v = 0;         % omega FIXED
c.logitthmu_v = 0.25;   c.logitthsa_v = 1;   % theta FREE

% Gamma parameter
% logit(gamma) ~ N(0, 1) => gamma prior mean ~ 0.5 in native space
c.logitgammamu = 0;
c.logitgammasa = 1;   % FREE (set to 0 to fix gamma)

% Gather prior settings in vectors
% Order: [14 standard params, gamma]
c.priormus = [
    c.mu2f_0mu, c.logsa2f_0mu, c.mu3f_0mu, c.logsa3f_0mu,...
    c.logitkamu_f, c.ommu_f, c.logitthmu_f,...
    c.mu2v_0mu, c.logsa2v_0mu, c.mu3v_0mu, c.logsa3v_0mu,...
    c.logitkamu_v, c.ommu_v, c.logitthmu_v,...
    c.logitgammamu,...
    ];

c.priorsas = [
    c.mu2f_0sa, c.logsa2f_0sa, c.mu3f_0sa, c.logsa3f_0sa,...
    c.logitkasa_f, c.omsa_f, c.logitthsa_f,...
    c.mu2v_0sa, c.logsa2v_0sa, c.mu3v_0sa, c.logsa3v_0sa,...
    c.logitkasa_v, c.omsa_v, c.logitthsa_v,...
    c.logitgammasa,...
    ];

c.prc_fun = @hgf_binary3l_facial_verbal_gamma;
c.transp_prc_fun = @hgf_binary3l_facial_verbal_gamma_transp;

return;
