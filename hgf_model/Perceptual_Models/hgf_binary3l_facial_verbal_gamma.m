function [traj, infStates] = hgf_binary3l_facial_verbal_gamma(r, p, varargin)
% M4 (Generalisation strength). Arch A extended with a free gamma in [0,1]
% that scales the EG-specific Level-2 prediction error before it updates
% the shared facial belief:
%   mu2_f(k) = mu2_f(k-1) + gamma * (1/pi2_f(k)) * da1_f(k)
% gamma -> 1 recovers Arch A (full generalisation); gamma -> 0 recovers
% per-EG independence behaviourally. Parameter vector = [Arch A, gamma].

% Transform parameters back to their native space if needed
if ~isempty(varargin) && strcmp(varargin{1},'trans');
    p = hgf_binary3l_facial_verbal_gamma_transp(r, p);
end

% Number of levels
l = 3;

% Unpack parameters — facial stream
mu2f_0 = p(1);
sa2f_0 = p(2);
mu3f_0 = p(3);
sa3f_0 = p(4);
ka_f   = p(5);
om_f   = p(6);
th_f   = p(7);

% Unpack parameters — verbal stream
mu2v_0 = p(8);
sa2v_0 = p(9);
mu3v_0 = p(10);
sa3v_0 = p(11);
ka_v   = p(12);
om_v   = p(13);
th_v   = p(14);

% Unpack gamma — NEW PARAMETER
gamma  = p(15);

% Add dummy "zeroth" trial
u_f = [0; r.u(:,1)];  % facial cue validity (binary)
u_v = [0; r.u(:,2)];  % verbal cue validity (binary)

% Number of trials (including prior)
n_f = length(u_f);
n_v = length(u_v);

% Time steps
t_f = ones(n_f,1);
t_v = t_f;

% Initialize updated quantities — facial
mu1_f = NaN(n_f,1);
mu2_f = NaN(n_f,1);
pi2_f = NaN(n_f,1);
mu3_f = NaN(n_f,1);
pi3_f = NaN(n_f,1);

% Initialize updated quantities — verbal
mu1_v = NaN(n_v,1);
mu2_v = NaN(n_v,1);
pi2_v = NaN(n_v,1);
mu3_v = NaN(n_v,1);
pi3_v = NaN(n_v,1);

% Other quantities — facial
mu1hat_f = NaN(n_f,1);
pi1hat_f = NaN(n_f,1);
pi2hat_f = NaN(n_f,1);
pi3hat_f = NaN(n_f,1);
w2_f     = NaN(n_f,1);
da1_f    = NaN(n_f,1);
da2_f    = NaN(n_f,1);

% Other quantities — verbal
mu1hat_v = NaN(n_v,1);
pi1hat_v = NaN(n_v,1);
pi2hat_v = NaN(n_v,1);
pi3hat_v = NaN(n_v,1);
w2_v     = NaN(n_v,1);
da1_v    = NaN(n_v,1);
da2_v    = NaN(n_v,1);

% Representation priors
mu1_f(1) = tapas_sgm(mu2f_0, 1);
mu2_f(1) = mu2f_0;
pi2_f(1) = 1/sa2f_0;
mu3_f(1) = mu3f_0;
pi3_f(1) = 1/sa3f_0;

mu1_v(1) = tapas_sgm(mu2v_0, 1);
mu2_v(1) = mu2v_0;
pi2_v(1) = 1/sa2v_0;
mu3_v(1) = mu3v_0;
pi3_v(1) = 1/sa3v_0;

% Pass through representation update loop
for k = 2:1:n_f
    if not(ismember(k-1, r.ign))

        %%%%%%%%%%%%%%%%%%%%%%
        % Effect of input u(k)
        %%%%%%%%%%%%%%%%%%%%%%

        % 1st level
        mu1hat_f(k) = tapas_sgm(mu2_f(k-1), 1);
        mu1hat_v(k) = tapas_sgm(mu2_v(k-1), 1);

        pi1hat_f(k) = 1/(mu1hat_f(k)*(1 -mu1hat_f(k)));
        pi1hat_v(k) = 1/(mu1hat_v(k)*(1 -mu1hat_v(k)));

        mu1_f(k) = u_f(k);
        mu1_v(k) = u_v(k);

        da1_f(k) = mu1_f(k) -mu1hat_f(k);
        da1_v(k) = mu1_v(k) -mu1hat_v(k);

        % 2nd level
        pi2hat_f(k) = 1/(1/pi2_f(k-1) +t_f(k) *exp(ka_f *mu3_f(k-1) +om_f));
        pi2hat_v(k) = 1/(1/pi2_v(k-1) +t_v(k) *exp(ka_v *mu3_v(k-1) +om_v));

        pi2_f(k) = pi2hat_f(k) +1/pi1hat_f(k);
        pi2_v(k) = pi2hat_v(k) +1/pi1hat_v(k);

        % *** KEY CHANGE: gamma modulates facial PE update ***
        mu2_f(k) = mu2_f(k-1) + gamma * 1/pi2_f(k) *da1_f(k);
        % Verbal stream: no gamma (always fully updated)
        mu2_v(k) = mu2_v(k-1) +1/pi2_v(k) *da1_v(k);

        % Volatility prediction error
        da2_f(k) = (1/pi2_f(k) +(mu2_f(k) -mu2_f(k-1))^2) *pi2hat_f(k) -1;
        da2_v(k) = (1/pi2_v(k) +(mu2_v(k) -mu2_v(k-1))^2) *pi2hat_v(k) -1;

        % 3rd level
        pi3hat_f(k) = 1/(1/pi3_f(k-1) +t_f(k) *th_f);
        pi3hat_v(k) = 1/(1/pi3_v(k-1) +t_v(k) *th_v);

        w2_f(k) = t_f(k) *exp(ka_f *mu3_f(k-1) +om_f) *pi2hat_f(k);
        w2_v(k) = t_v(k) *exp(ka_v *mu3_v(k-1) +om_v) *pi2hat_v(k);

        pi3_f(k) = pi3hat_f(k) +1/2 *ka_f^2 *w2_f(k) *(w2_f(k) +(2 *w2_f(k) -1) *da2_f(k));
        pi3_v(k) = pi3hat_v(k) +1/2 *ka_v^2 *w2_v(k) *(w2_v(k) +(2 *w2_v(k) -1) *da2_v(k));

        if pi3_f(k) <= 0
            error('Error: negative pi3 in facial learning.');
        end
        if pi3_v(k) <= 0
            error('Error: negative pi3 in verbal learning.');
        end

        mu3_f(k) = mu3_f(k-1) +1/2 *1/pi3_f(k) *ka_f *w2_f(k) *da2_f(k);
        mu3_v(k) = mu3_v(k-1) +1/2 *1/pi3_v(k) *ka_v *w2_v(k) *da2_v(k);

    else
        mu1_f(k) = mu1_f(k-1);
        mu2_f(k) = mu2_f(k-1);
        pi2_f(k) = pi2_f(k-1);
        mu3_f(k) = mu3_f(k-1);
        pi3_f(k) = pi3_f(k-1);
        mu1hat_f(k) = mu1hat_f(k-1);
        pi1hat_f(k) = pi1hat_f(k-1);
        pi2hat_f(k) = pi2hat_f(k-1);
        pi3hat_f(k) = pi3hat_f(k-1);
        w2_f(k)     = w2_f(k-1);
        da1_f(k)    = da1_f(k-1);
        da2_f(k)    = da2_f(k-1);

        mu1_v(k) = mu1_v(k-1);
        mu2_v(k) = mu2_v(k-1);
        pi2_v(k) = pi2_v(k-1);
        mu3_v(k) = mu3_v(k-1);
        pi3_v(k) = pi3_v(k-1);
        mu1hat_v(k) = mu1hat_v(k-1);
        pi1hat_v(k) = pi1hat_v(k-1);
        pi2hat_v(k) = pi2hat_v(k-1);
        pi3hat_v(k) = pi3hat_v(k-1);
        w2_v(k)     = w2_v(k-1);
        da1_v(k)    = da1_v(k-1);
        da2_v(k)    = da2_v(k-1);
    end
end

% Get predictions
mu2hat_f = mu2_f; mu2hat_f(end) = [];
mu3hat_f = mu3_f; mu3hat_f(end) = [];
mu2hat_v = mu2_v; mu2hat_v(end) = [];
mu3hat_v = mu3_v; mu3hat_v(end) = [];

% Remove priors
mu1_f(1)=[]; mu2_f(1)=[]; pi2_f(1)=[]; mu3_f(1)=[]; pi3_f(1)=[];
mu1_v(1)=[]; mu2_v(1)=[]; pi2_v(1)=[]; mu3_v(1)=[]; pi3_v(1)=[];
mu1hat_f(1)=[]; pi1hat_f(1)=[]; pi2hat_f(1)=[]; pi3hat_f(1)=[];
w2_f(1)=[]; da1_f(1)=[]; da2_f(1)=[];
mu1hat_v(1)=[]; pi1hat_v(1)=[]; pi2hat_v(1)=[]; pi3hat_v(1)=[];
w2_v(1)=[]; da1_v(1)=[]; da2_v(1)=[];

% Variance at 1st level
sa1_f = mu1_f.*(1-mu1_f);
sa1_v = mu1_v.*(1-mu1_v);

% Create result data structure
traj = struct;
traj.mu_f = [mu1_f, mu2_f, mu3_f];
traj.sa_f = [sa1_f, 1./pi2_f, 1./pi3_f];
traj.mu_v = [mu1_v, mu2_v, mu3_v];
traj.sa_v = [sa1_v, 1./pi2_v, 1./pi3_v];
traj.muhat_f = [mu1hat_f, mu2hat_f, mu3hat_f];
traj.sahat_f = [1./pi1hat_f, 1./pi2hat_f, 1./pi3hat_f];
traj.muhat_v = [mu1hat_v, mu2hat_v, mu3hat_v];
traj.sahat_v = [1./pi1hat_v, 1./pi2hat_v, 1./pi3hat_v];
traj.w_f = w2_f; traj.da_f = [da1_f, da2_f];
traj.w_v = w2_v; traj.da_v = [da1_v, da2_v];

% Matrices for observation model
infStates = NaN(n_f-1,l,4);
infStates(:,:,1) = traj.muhat_f;
infStates(:,:,2) = traj.sahat_f;
infStates(:,:,3) = traj.muhat_v;
infStates(:,:,4) = traj.sahat_v;

return;
