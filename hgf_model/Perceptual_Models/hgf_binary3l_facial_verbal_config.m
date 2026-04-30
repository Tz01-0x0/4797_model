function c = hgf_binary3l_facial_verbal_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configuration for the 3-level binary HGF for facial and verbal cue learning.
% Architecture A: Two parallel HGFs (facial + verbal), omega FIXED.
%
% Adapted from hgf_binary3l_reward_social_fixOmega_config.m
% Winning model configuration from Sevgi et al. with omega fixed at -4.
%
% Free parameters: kappa_f, theta_f, kappa_v, theta_v (4 perceptual params)
% Fixed parameters: omega_f = omega_v = -4, all initial conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Config structure
c = struct;

% Model name
c.model = 'hgf_binary3l_facial_verbal';

% Upper bound for kappa and theta (lower bound is always zero)
c.kaub_f = 1;
c.thub_f = 1;
c.kaub_v = 1;
c.thub_v = 1;

% Sufficient statistics of Gaussian parameter priors

% Initial mu2 — fixed to 0 (neutral)
c.mu2f_0mu = 0;
c.mu2f_0sa = 0;
c.mu2v_0mu = 0;
c.mu2v_0sa = 0;

% Initial sigma2 — fixed
c.logsa2f_0mu = log(1);
c.logsa2f_0sa = 0;
c.logsa2v_0mu = log(1);
c.logsa2v_0sa = 0;

% Initial mu3 — fixed to 1
c.mu3f_0mu = 1;
c.mu3f_0sa = 0;
c.mu3v_0mu = 1;
c.mu3v_0sa = 0;

% Initial sigma3 — fixed
c.logsa3f_0mu = log(1);
c.logsa3f_0sa = 0;
c.logsa3v_0mu = log(1);
c.logsa3v_0sa = 0;

% Kappa — FREE
c.logitkamu_f = 0;
c.logitkasa_f = 1;
c.logitkamu_v = 0;
c.logitkasa_v = 1;

% Omega — FIXED at -4 (winning model from Sevgi et al.)
c.ommu_f = -4;
c.omsa_f = 0;
c.ommu_v = -4;
c.omsa_v = 0;

% Theta — FREE
c.logitthmu_f = 0.25;
c.logitthsa_f = 1;
c.logitthmu_v = 0.25;
c.logitthsa_v = 1;

% Gather prior settings in vectors
% Parameter order: [mu2f_0, sa2f_0, mu3f_0, sa3f_0, ka_f, om_f, th_f,
%                   mu2v_0, sa2v_0, mu3v_0, sa3v_0, ka_v, om_v, th_v]
c.priormus = [
    c.mu2f_0mu,...
    c.logsa2f_0mu,...
    c.mu3f_0mu,...
    c.logsa3f_0mu,...
    c.logitkamu_f,...
    c.ommu_f,...
    c.logitthmu_f,...
    c.mu2v_0mu,...
    c.logsa2v_0mu,...
    c.mu3v_0mu,...
    c.logsa3v_0mu,...
    c.logitkamu_v,...
    c.ommu_v,...
    c.logitthmu_v,...
    ];

c.priorsas = [
    c.mu2f_0sa,...
    c.logsa2f_0sa,...
    c.mu3f_0sa,...
    c.logsa3f_0sa,...
    c.logitkasa_f,...
    c.omsa_f,...
    c.logitthsa_f,...
    c.mu2v_0sa,...
    c.logsa2v_0sa,...
    c.mu3v_0sa,...
    c.logsa3v_0sa,...
    c.logitkasa_v,...
    c.omsa_v,...
    c.logitthsa_v,...
    ];

% Model function handle
c.prc_fun = @hgf_binary3l_facial_verbal;

% Handle to function that transforms perceptual parameters to their native space
c.transp_prc_fun = @hgf_binary3l_facial_verbal_transp;

return;
