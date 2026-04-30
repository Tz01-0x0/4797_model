function [traj, infStates] = hgf_binary3l_facial_verbal_archB(r, p, varargin)
% Architecture B (Separated). Three independent facial HGFs (one per EG:
% 1=A, 2=B, 3=C) sharing kappa_f / omega_f / theta_f. Per trial, only the
% facial HGF whose EG is presented (col 7 of r.u) is updated; the others
% carry their state forward. Verbal stream unchanged from Arch A. Parameter
% vector matches Arch A; outputs expose the currently-active EG's trajectory
% so the softmax response model is reused unmodified.

% Transform parameters back to their native space if needed
if ~isempty(varargin) && strcmp(varargin{1},'trans')
    p = hgf_binary3l_facial_verbal_transp(r, p);
end

% Number of levels
l = 3;
nG = 3;  % three equivalence groups

% Unpack parameters — facial stream (shared across groups)
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

% Add dummy "zeroth" trial
u_f = [0; r.u(:,1)];     % facial cue validity
u_v = [0; r.u(:,2)];     % verbal cue validity
g   = [1; r.u(:,7)];     % group index per trial (dummy group=1 for trial 0)

n = length(u_f);

% Time steps
t = ones(n,1);

% Per-group facial trajectories

% Indexed as (trial, group)
mu1_fG = NaN(n,nG);
mu2_fG = NaN(n,nG);
pi2_fG = NaN(n,nG);
mu3_fG = NaN(n,nG);
pi3_fG = NaN(n,nG);

mu1hat_fG = NaN(n,nG);
pi1hat_fG = NaN(n,nG);
pi2hat_fG = NaN(n,nG);
pi3hat_fG = NaN(n,nG);
w2_fG     = NaN(n,nG);
da1_fG    = NaN(n,nG);
da2_fG    = NaN(n,nG);

% Priors for each facial HGF
for gi = 1:nG
    mu1_fG(1,gi) = tapas_sgm(mu2f_0, 1);
    mu2_fG(1,gi) = mu2f_0;
    pi2_fG(1,gi) = 1/sa2f_0;
    mu3_fG(1,gi) = mu3f_0;
    pi3_fG(1,gi) = 1/sa3f_0;
end

% Verbal trajectories (unchanged)

mu1_v = NaN(n,1);
mu2_v = NaN(n,1);
pi2_v = NaN(n,1);
mu3_v = NaN(n,1);
pi3_v = NaN(n,1);

mu1hat_v = NaN(n,1);
pi1hat_v = NaN(n,1);
pi2hat_v = NaN(n,1);
pi3hat_v = NaN(n,1);
w2_v     = NaN(n,1);
da1_v    = NaN(n,1);
da2_v    = NaN(n,1);

mu1_v(1) = tapas_sgm(mu2v_0, 1);
mu2_v(1) = mu2v_0;
pi2_v(1) = 1/sa2v_0;
mu3_v(1) = mu3v_0;
pi3_v(1) = 1/sa3v_0;

% Pass through representation update loop
for k = 2:n
    if not(ismember(k-1, r.ign))
        gk = g(k);

        % Facial: update only the active group
        for gi = 1:nG
            if gi == gk
                % Predict
                mu1hat_fG(k,gi) = tapas_sgm(mu2_fG(k-1,gi), 1);
                pi1hat_fG(k,gi) = 1/(mu1hat_fG(k,gi)*(1-mu1hat_fG(k,gi)));

                % Input
                mu1_fG(k,gi) = u_f(k);
                da1_fG(k,gi) = mu1_fG(k,gi) - mu1hat_fG(k,gi);

                % 2nd level
                pi2hat_fG(k,gi) = 1/(1/pi2_fG(k-1,gi) + t(k)*exp(ka_f*mu3_fG(k-1,gi) + om_f));
                pi2_fG(k,gi)    = pi2hat_fG(k,gi) + 1/pi1hat_fG(k,gi);
                mu2_fG(k,gi)    = mu2_fG(k-1,gi) + 1/pi2_fG(k,gi) * da1_fG(k,gi);

                % Volatility PE
                da2_fG(k,gi) = (1/pi2_fG(k,gi) + (mu2_fG(k,gi)-mu2_fG(k-1,gi))^2) * pi2hat_fG(k,gi) - 1;

                % 3rd level
                pi3hat_fG(k,gi) = 1/(1/pi3_fG(k-1,gi) + t(k)*th_f);
                w2_fG(k,gi)     = t(k)*exp(ka_f*mu3_fG(k-1,gi) + om_f) * pi2hat_fG(k,gi);
                pi3_fG(k,gi)    = pi3hat_fG(k,gi) + 1/2 * ka_f^2 * w2_fG(k,gi) * (w2_fG(k,gi) + (2*w2_fG(k,gi)-1)*da2_fG(k,gi));

                if pi3_fG(k,gi) <= 0
                    error('Error: negative pi3 in facial learning (archB, group %d).', gi);
                end

                mu3_fG(k,gi) = mu3_fG(k-1,gi) + 1/2 * 1/pi3_fG(k,gi) * ka_f * w2_fG(k,gi) * da2_fG(k,gi);
            else
                % Carry forward inactive groups
                mu1_fG(k,gi) = mu1_fG(k-1,gi);
                mu2_fG(k,gi) = mu2_fG(k-1,gi);
                pi2_fG(k,gi) = pi2_fG(k-1,gi);
                mu3_fG(k,gi) = mu3_fG(k-1,gi);
                pi3_fG(k,gi) = pi3_fG(k-1,gi);

                mu1hat_fG(k,gi) = mu1hat_fG(k-1,gi);
                pi1hat_fG(k,gi) = pi1hat_fG(k-1,gi);
                pi2hat_fG(k,gi) = pi2hat_fG(k-1,gi);
                pi3hat_fG(k,gi) = pi3hat_fG(k-1,gi);
                w2_fG(k,gi)     = w2_fG(k-1,gi);
                da1_fG(k,gi)    = da1_fG(k-1,gi);
                da2_fG(k,gi)    = da2_fG(k-1,gi);
            end
        end

        % Verbal: always update
        mu1hat_v(k) = tapas_sgm(mu2_v(k-1), 1);
        pi1hat_v(k) = 1/(mu1hat_v(k)*(1-mu1hat_v(k)));
        mu1_v(k)    = u_v(k);
        da1_v(k)    = mu1_v(k) - mu1hat_v(k);

        pi2hat_v(k) = 1/(1/pi2_v(k-1) + t(k)*exp(ka_v*mu3_v(k-1) + om_v));
        pi2_v(k)    = pi2hat_v(k) + 1/pi1hat_v(k);
        mu2_v(k)    = mu2_v(k-1) + 1/pi2_v(k) * da1_v(k);

        da2_v(k) = (1/pi2_v(k) + (mu2_v(k)-mu2_v(k-1))^2) * pi2hat_v(k) - 1;

        pi3hat_v(k) = 1/(1/pi3_v(k-1) + t(k)*th_v);
        w2_v(k)     = t(k)*exp(ka_v*mu3_v(k-1) + om_v) * pi2hat_v(k);
        pi3_v(k)    = pi3hat_v(k) + 1/2 * ka_v^2 * w2_v(k) * (w2_v(k) + (2*w2_v(k)-1)*da2_v(k));

        if pi3_v(k) <= 0
            error('Error: negative pi3 in verbal learning (archB).');
        end

        mu3_v(k) = mu3_v(k-1) + 1/2 * 1/pi3_v(k) * ka_v * w2_v(k) * da2_v(k);

    else
        % Ignored trial — carry everything forward
        for gi = 1:nG
            mu1_fG(k,gi) = mu1_fG(k-1,gi);
            mu2_fG(k,gi) = mu2_fG(k-1,gi);
            pi2_fG(k,gi) = pi2_fG(k-1,gi);
            mu3_fG(k,gi) = mu3_fG(k-1,gi);
            pi3_fG(k,gi) = pi3_fG(k-1,gi);

            mu1hat_fG(k,gi) = mu1hat_fG(k-1,gi);
            pi1hat_fG(k,gi) = pi1hat_fG(k-1,gi);
            pi2hat_fG(k,gi) = pi2hat_fG(k-1,gi);
            pi3hat_fG(k,gi) = pi3hat_fG(k-1,gi);
            w2_fG(k,gi)     = w2_fG(k-1,gi);
            da1_fG(k,gi)    = da1_fG(k-1,gi);
            da2_fG(k,gi)    = da2_fG(k-1,gi);
        end

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

% Assemble active-group facial trajectories (for response model)
% - On each trial, pick the row corresponding to the group presented.
mu1_f   = NaN(n,1); mu2_f   = NaN(n,1); mu3_f   = NaN(n,1);
pi2_f   = NaN(n,1); pi3_f   = NaN(n,1);
mu1hat_f = NaN(n,1); pi1hat_f = NaN(n,1);
pi2hat_f = NaN(n,1); pi3hat_f = NaN(n,1);
w2_f     = NaN(n,1); da1_f    = NaN(n,1); da2_f = NaN(n,1);

for k = 1:n
    gk = g(k);
    mu1_f(k)   = mu1_fG(k,gk);
    mu2_f(k)   = mu2_fG(k,gk);
    mu3_f(k)   = mu3_fG(k,gk);
    pi2_f(k)   = pi2_fG(k,gk);
    pi3_f(k)   = pi3_fG(k,gk);
    mu1hat_f(k) = mu1hat_fG(k,gk);
    pi1hat_f(k) = pi1hat_fG(k,gk);
    pi2hat_f(k) = pi2hat_fG(k,gk);
    pi3hat_f(k) = pi3hat_fG(k,gk);
    w2_f(k)     = w2_fG(k,gk);
    da1_f(k)    = da1_fG(k,gk);
    da2_f(k)    = da2_fG(k,gk);
end

% muhat/sahat for ArchA-compatible output: take state at k-1 reading as prediction
mu2hat_f = mu2_f; mu2hat_f(end) = [];
mu3hat_f = mu3_f; mu3hat_f(end) = [];
mu2hat_v = mu2_v; mu2hat_v(end) = [];
mu3hat_v = mu3_v; mu3hat_v(end) = [];

% Remove priors (trial 0)
mu1_f(1)=[]; mu2_f(1)=[]; pi2_f(1)=[]; mu3_f(1)=[]; pi3_f(1)=[];
mu1_v(1)=[]; mu2_v(1)=[]; pi2_v(1)=[]; mu3_v(1)=[]; pi3_v(1)=[];
mu1hat_f(1)=[]; pi1hat_f(1)=[]; pi2hat_f(1)=[]; pi3hat_f(1)=[];
w2_f(1)=[]; da1_f(1)=[]; da2_f(1)=[];
mu1hat_v(1)=[]; pi1hat_v(1)=[]; pi2hat_v(1)=[]; pi3hat_v(1)=[];
w2_v(1)=[]; da1_v(1)=[]; da2_v(1)=[];

% Variance at 1st level
sa1_f = mu1_f.*(1-mu1_f);
sa1_v = mu1_v.*(1-mu1_v);

% Also keep per-group full trajectories for diagnostics
traj = struct;
traj.mu_f = [mu1_f, mu2_f, mu3_f];
traj.sa_f = [sa1_f, 1./pi2_f, 1./pi3_f];
traj.mu_v = [mu1_v, mu2_v, mu3_v];
traj.sa_v = [sa1_v, 1./pi2_v, 1./pi3_v];
traj.muhat_f = [mu1hat_f, mu2hat_f, mu3hat_f];
traj.sahat_f = [1./pi1hat_f, 1./pi2hat_f, 1./pi3hat_f];
traj.muhat_v = [mu1hat_v, mu2hat_v, mu3hat_v];
traj.sahat_v = [1./pi1hat_v, 1./pi2hat_v, 1./pi3hat_v];
traj.w_f  = w2_f;
traj.da_f = [da1_f, da2_f];
traj.w_v  = w2_v;
traj.da_v = [da1_v, da2_v];

% Per-group facial states (for diagnostics / future plotting)
traj.group_mu2_f = mu2_fG(2:end,:);
traj.group_mu3_f = mu3_fG(2:end,:);
traj.group_sa2_f = 1./pi2_fG(2:end,:);
traj.group_sa3_f = 1./pi3_fG(2:end,:);

% infStates for response model — active group facial stream
nOut = n - 1;
infStates = NaN(nOut, l, 4);
infStates(:,:,1) = traj.muhat_f;
infStates(:,:,2) = traj.sahat_f;
infStates(:,:,3) = traj.muhat_v;
infStates(:,:,4) = traj.sahat_v;

return;
