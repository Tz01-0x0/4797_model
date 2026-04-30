function [traj, infStates] = hgf_binary3l_facial_verbal_archC(r, p, varargin)
% Architecture C (Hybrid). Shared facial Level-3 volatility x3_f across all
% three EGs (1=A, 2=B, 3=C); separate Level-2 x2_f per EG to absorb the
% systematic accuracy gradient. The active EG's Level-2 prediction error
% drives the shared x3_f update. Parameter vector matches Arch A; verbal
% stream unchanged. Output exposes the active EG's facial trajectory.

if ~isempty(varargin) && strcmp(varargin{1},'trans')
    p = hgf_binary3l_facial_verbal_transp(r, p);
end

l = 3;
nG = 3;

% Facial parameters (shared)
mu2f_0 = p(1);
sa2f_0 = p(2);
mu3f_0 = p(3);
sa3f_0 = p(4);
ka_f   = p(5);
om_f   = p(6);
th_f   = p(7);

% Verbal parameters
mu2v_0 = p(8);
sa2v_0 = p(9);
mu3v_0 = p(10);
sa3v_0 = p(11);
ka_v   = p(12);
om_v   = p(13);
th_v   = p(14);

u_f = [0; r.u(:,1)];
u_v = [0; r.u(:,2)];
g   = [1; r.u(:,7)];

n = length(u_f);
t = ones(n,1);

% Per-group x1, x2 for facial
mu1_fG = NaN(n,nG);
mu2_fG = NaN(n,nG);
pi2_fG = NaN(n,nG);

mu1hat_fG = NaN(n,nG);
pi1hat_fG = NaN(n,nG);
pi2hat_fG = NaN(n,nG);
w2_fG     = NaN(n,nG);
da1_fG    = NaN(n,nG);
da2_fG    = NaN(n,nG);

% Shared x3 for facial
mu3_fS  = NaN(n,1);
pi3_fS  = NaN(n,1);
pi3hat_fS = NaN(n,1);

% Priors
for gi = 1:nG
    mu1_fG(1,gi) = tapas_sgm(mu2f_0, 1);
    mu2_fG(1,gi) = mu2f_0;
    pi2_fG(1,gi) = 1/sa2f_0;
end
mu3_fS(1) = mu3f_0;
pi3_fS(1) = 1/sa3f_0;

% Verbal (standard, unchanged)
mu1_v = NaN(n,1); mu2_v = NaN(n,1); pi2_v = NaN(n,1);
mu3_v = NaN(n,1); pi3_v = NaN(n,1);
mu1hat_v = NaN(n,1); pi1hat_v = NaN(n,1);
pi2hat_v = NaN(n,1); pi3hat_v = NaN(n,1);
w2_v = NaN(n,1); da1_v = NaN(n,1); da2_v = NaN(n,1);

mu1_v(1) = tapas_sgm(mu2v_0, 1);
mu2_v(1) = mu2v_0;
pi2_v(1) = 1/sa2v_0;
mu3_v(1) = mu3v_0;
pi3_v(1) = 1/sa3v_0;

for k = 2:n
    if not(ismember(k-1, r.ign))
        gk = g(k);

        % Facial, group-selective x1 & x2; shared x3
        for gi = 1:nG
            if gi == gk
                mu1hat_fG(k,gi) = tapas_sgm(mu2_fG(k-1,gi), 1);
                pi1hat_fG(k,gi) = 1/(mu1hat_fG(k,gi)*(1-mu1hat_fG(k,gi)));
                mu1_fG(k,gi) = u_f(k);
                da1_fG(k,gi) = mu1_fG(k,gi) - mu1hat_fG(k,gi);

                % 2nd level uses SHARED x3
                pi2hat_fG(k,gi) = 1/(1/pi2_fG(k-1,gi) + t(k)*exp(ka_f*mu3_fS(k-1) + om_f));
                pi2_fG(k,gi)    = pi2hat_fG(k,gi) + 1/pi1hat_fG(k,gi);
                mu2_fG(k,gi)    = mu2_fG(k-1,gi) + 1/pi2_fG(k,gi) * da1_fG(k,gi);

                da2_fG(k,gi) = (1/pi2_fG(k,gi) + (mu2_fG(k,gi)-mu2_fG(k-1,gi))^2) * pi2hat_fG(k,gi) - 1;
                w2_fG(k,gi)  = t(k)*exp(ka_f*mu3_fS(k-1) + om_f) * pi2hat_fG(k,gi);
            else
                % Inactive groups carry x2 forward
                mu1_fG(k,gi) = mu1_fG(k-1,gi);
                mu2_fG(k,gi) = mu2_fG(k-1,gi);
                pi2_fG(k,gi) = pi2_fG(k-1,gi);
                mu1hat_fG(k,gi) = mu1hat_fG(k-1,gi);
                pi1hat_fG(k,gi) = pi1hat_fG(k-1,gi);
                pi2hat_fG(k,gi) = pi2hat_fG(k-1,gi);
                w2_fG(k,gi)     = w2_fG(k-1,gi);
                da1_fG(k,gi)    = da1_fG(k-1,gi);
                da2_fG(k,gi)    = da2_fG(k-1,gi);
            end
        end

        % Shared x3 update driven by active group's da2
        pi3hat_fS(k) = 1/(1/pi3_fS(k-1) + t(k)*th_f);
        w2_active    = w2_fG(k,gk);
        da2_active   = da2_fG(k,gk);
        pi3_fS(k)    = pi3hat_fS(k) + 1/2 * ka_f^2 * w2_active * (w2_active + (2*w2_active-1)*da2_active);
        if pi3_fS(k) <= 0
            error('Error: negative pi3 in facial learning (archC shared).');
        end
        mu3_fS(k) = mu3_fS(k-1) + 1/2 * 1/pi3_fS(k) * ka_f * w2_active * da2_active;

        % Verbal
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
            error('Error: negative pi3 in verbal learning (archC).');
        end
        mu3_v(k) = mu3_v(k-1) + 1/2 * 1/pi3_v(k) * ka_v * w2_v(k) * da2_v(k);

    else
        % Ignored trial — carry all forward
        for gi = 1:nG
            mu1_fG(k,gi) = mu1_fG(k-1,gi);
            mu2_fG(k,gi) = mu2_fG(k-1,gi);
            pi2_fG(k,gi) = pi2_fG(k-1,gi);
            mu1hat_fG(k,gi) = mu1hat_fG(k-1,gi);
            pi1hat_fG(k,gi) = pi1hat_fG(k-1,gi);
            pi2hat_fG(k,gi) = pi2hat_fG(k-1,gi);
            w2_fG(k,gi)     = w2_fG(k-1,gi);
            da1_fG(k,gi)    = da1_fG(k-1,gi);
            da2_fG(k,gi)    = da2_fG(k-1,gi);
        end
        mu3_fS(k)    = mu3_fS(k-1);
        pi3_fS(k)    = pi3_fS(k-1);
        pi3hat_fS(k) = pi3hat_fS(k-1);

        mu1_v(k) = mu1_v(k-1); mu2_v(k) = mu2_v(k-1); pi2_v(k) = pi2_v(k-1);
        mu3_v(k) = mu3_v(k-1); pi3_v(k) = pi3_v(k-1);
        mu1hat_v(k) = mu1hat_v(k-1); pi1hat_v(k) = pi1hat_v(k-1);
        pi2hat_v(k) = pi2hat_v(k-1); pi3hat_v(k) = pi3hat_v(k-1);
        w2_v(k) = w2_v(k-1); da1_v(k) = da1_v(k-1); da2_v(k) = da2_v(k-1);
    end
end

% Assemble active-group facial outputs
mu1_f = NaN(n,1); mu2_f = NaN(n,1);
pi2_f = NaN(n,1);
mu1hat_f = NaN(n,1); pi1hat_f = NaN(n,1); pi2hat_f = NaN(n,1);
w2_f = NaN(n,1); da1_f = NaN(n,1); da2_f = NaN(n,1);

for k = 1:n
    gk = g(k);
    mu1_f(k)    = mu1_fG(k,gk);
    mu2_f(k)    = mu2_fG(k,gk);
    pi2_f(k)    = pi2_fG(k,gk);
    mu1hat_f(k) = mu1hat_fG(k,gk);
    pi1hat_f(k) = pi1hat_fG(k,gk);
    pi2hat_f(k) = pi2hat_fG(k,gk);
    w2_f(k)     = w2_fG(k,gk);
    da1_f(k)    = da1_fG(k,gk);
    da2_f(k)    = da2_fG(k,gk);
end

% Shared x3 replicated as the facial L3 trajectory
mu3_f = mu3_fS;
pi3_f = pi3_fS;

% Predictions (mu at k-1)
mu2hat_f = mu2_f; mu2hat_f(end) = [];
mu3hat_f = mu3_f; mu3hat_f(end) = [];
mu2hat_v = mu2_v; mu2hat_v(end) = [];
mu3hat_v = mu3_v; mu3hat_v(end) = [];

% Remove trial-0
mu1_f(1)=[]; mu2_f(1)=[]; pi2_f(1)=[]; mu3_f(1)=[]; pi3_f(1)=[];
mu1_v(1)=[]; mu2_v(1)=[]; pi2_v(1)=[]; mu3_v(1)=[]; pi3_v(1)=[];
mu1hat_f(1)=[]; pi1hat_f(1)=[]; pi2hat_f(1)=[];
w2_f(1)=[]; da1_f(1)=[]; da2_f(1)=[];
pi3hat_fS(1)=[];
mu1hat_v(1)=[]; pi1hat_v(1)=[]; pi2hat_v(1)=[]; pi3hat_v(1)=[];
w2_v(1)=[]; da1_v(1)=[]; da2_v(1)=[];

% pi3hat_f for active-group trials = shared pi3hat_fS
pi3hat_f = pi3hat_fS;

sa1_f = mu1_f.*(1-mu1_f);
sa1_v = mu1_v.*(1-mu1_v);

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

% Per-group x2 trajectories for diagnostics
traj.group_mu2_f = mu2_fG(2:end,:);
traj.group_sa2_f = 1./pi2_fG(2:end,:);
traj.shared_mu3_f = mu3_fS(2:end);
traj.shared_sa3_f = 1./pi3_fS(2:end);

nOut = n - 1;
infStates = NaN(nOut, l, 4);
infStates(:,:,1) = traj.muhat_f;
infStates(:,:,2) = traj.sahat_f;
infStates(:,:,3) = traj.muhat_v;
infStates(:,:,4) = traj.sahat_v;

return;
