function c = hgf_binary3l_facial_verbal_I1mean_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I1-mean variant: Architecture A with FREE initial mu2_0 for both streams.
%
% Tests whether allowing different initial BELIEFS (mean of level-2)
% for facial vs verbal streams improves model fit.
%
% Free parameters: mu2f_0, mu2v_0, kappa_f, theta_f, kappa_v, theta_v (6 perceptual)
% Fixed: sa2_0 = 1, mu3_0 = 1, sa3_0 = 1, omega = -4 (both streams)
%
% Rationale: If ASD participants start with different prior expectations
% about facial vs verbal cue reliability, this would be captured by
% differential mu2_0 values. This tests "mean pathway" dissociation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = struct;
c.model = 'hgf_binary3l_facial_verbal';  % Reuse same implementation

% Upper bounds
c.kaub_f = 1;
c.thub_f = 1;
c.kaub_v = 1;
c.thub_v = 1;

% Facial stream

% mu2f_0: FREE — initial belief about facial cue reliability
c.mu2f_0mu = 0;
c.mu2f_0sa = 1;        % FREE (was 0)

% sa2f_0: FIXED at 1
c.logsa2f_0mu = log(1);
c.logsa2f_0sa = 0;

% mu3f_0: FIXED at 1
c.mu3f_0mu = 1;
c.mu3f_0sa = 0;

% sa3f_0: FIXED at 1
c.logsa3f_0mu = log(1);
c.logsa3f_0sa = 0;

% kappa_f: FREE
c.logitkamu_f = 0;
c.logitkasa_f = 1;

% omega_f: FIXED at -4
c.ommu_f = -4;
c.omsa_f = 0;

% theta_f: FREE
c.logitthmu_f = 0.25;
c.logitthsa_f = 1;

% Verbal stream

% mu2v_0: FREE — initial belief about verbal cue reliability
c.mu2v_0mu = 0;
c.mu2v_0sa = 1;        % FREE (was 0)

% sa2v_0: FIXED at 1
c.logsa2v_0mu = log(1);
c.logsa2v_0sa = 0;

% mu3v_0: FIXED at 1
c.mu3v_0mu = 1;
c.mu3v_0sa = 0;

% sa3v_0: FIXED at 1
c.logsa3v_0mu = log(1);
c.logsa3v_0sa = 0;

% kappa_v: FREE
c.logitkamu_v = 0;
c.logitkasa_v = 1;

% omega_v: FIXED at -4
c.ommu_v = -4;
c.omsa_v = 0;

% theta_v: FREE
c.logitthmu_v = 0.25;
c.logitthsa_v = 1;

% Gather prior settings
c.priormus = [
    c.mu2f_0mu, c.logsa2f_0mu, c.mu3f_0mu, c.logsa3f_0mu,...
    c.logitkamu_f, c.ommu_f, c.logitthmu_f,...
    c.mu2v_0mu, c.logsa2v_0mu, c.mu3v_0mu, c.logsa3v_0mu,...
    c.logitkamu_v, c.ommu_v, c.logitthmu_v,...
    ];

c.priorsas = [
    c.mu2f_0sa, c.logsa2f_0sa, c.mu3f_0sa, c.logsa3f_0sa,...
    c.logitkasa_f, c.omsa_f, c.logitthsa_f,...
    c.mu2v_0sa, c.logsa2v_0sa, c.mu3v_0sa, c.logsa3v_0sa,...
    c.logitkasa_v, c.omsa_v, c.logitthsa_v,...
    ];

c.prc_fun = @hgf_binary3l_facial_verbal;
c.transp_prc_fun = @hgf_binary3l_facial_verbal_transp;

return;
