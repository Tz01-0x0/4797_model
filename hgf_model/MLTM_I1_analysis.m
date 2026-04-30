function MLTM_I1_analysis()
% Group comparison and AQ correlations for the I1_Both initial-condition
% variant (free mu2_0 and sa2_0 on both cue streams). Pulls fits from
% results_I1/ and writes a summary alongside.

%% Setup
options = MLTM_options_new();
resultdir = fullfile(options.code, 'results_I1');
meta = readtable(options.metadata);

prc_config = 'hgf_binary3l_facial_verbal_I1both_config';
obs_config = 'softmax_facial_verbal_config';

nSubj = length(options.subjects);

% Preallocate
mu2f_0 = NaN(nSubj, 1);
sa2f_0 = NaN(nSubj, 1);
mu2v_0 = NaN(nSubj, 1);
sa2v_0 = NaN(nSubj, 1);
kappaF = NaN(nSubj, 1);
thetaF = NaN(nSubj, 1);
kappaV = NaN(nSubj, 1);
thetaV = NaN(nSubj, 1);
zeta   = NaN(nSubj, 1);
beta   = NaN(nSubj, 1);
group  = cell(nSubj, 1);
aq     = NaN(nSubj, 1);
condition = cell(nSubj, 1);

fprintf('\n=== I1_Both Parameter Extraction ===\n');
nLoaded = 0;

for s = 1:nSubj
    sid = options.subjects{s};
    fname = fullfile(resultdir, sprintf('%s_%s_%s.mat', sid, prc_config, obs_config));

    if ~exist(fname, 'file')
        fprintf('  WARNING: Missing %s\n', sid);
        continue;
    end

    est_struct = load(fname);
    if ~isfield(est_struct, 'est') || ~isfield(est_struct.est, 'p_prc')
        fprintf('  WARNING: Invalid structure for %s\n', sid);
        continue;
    end

    p_prc = est_struct.est.p_prc;
    p_obs = est_struct.est.p_obs;

    % Initial conditions (I1_Both specific)
    mu2f_0(s) = p_prc.mu2f_0;
    sa2f_0(s) = p_prc.sa2f_0;
    mu2v_0(s) = p_prc.mu2v_0;
    sa2v_0(s) = p_prc.sa2v_0;

    % Standard HGF parameters
    kappaF(s) = p_prc.ka_f;
    thetaF(s) = p_prc.th_f;
    kappaV(s) = p_prc.ka_v;
    thetaV(s) = p_prc.th_v;

    % Response model parameters
    zeta(s) = p_obs.ze1;
    beta(s) = p_obs.beta;

    % Group and AQ from metadata
    idx = find(strcmp(meta.participant_id, sid));
    if ~isempty(idx)
        group{s} = meta.group{idx};
        if ismember('aq_total', meta.Properties.VariableNames)
            aq(s) = meta.aq_total(idx);
        end
        if ismember('randomiser_ne38', meta.Properties.VariableNames)
            condition{s} = char(meta.randomiser_ne38(idx));
        elseif ismember('condition', meta.Properties.VariableNames)
            condition{s} = char(meta.condition(idx));
        end
    else
        group{s} = '?';
        condition{s} = '?';
    end

    nLoaded = nLoaded + 1;
end

fprintf('Loaded %d / %d subjects\n\n', nLoaded, nSubj);

%% Save full CSV
T = table(options.subjects(:), group, condition, ...
    mu2f_0, sa2f_0, mu2v_0, sa2v_0, ...
    kappaF, thetaF, kappaV, thetaV, zeta, beta, aq, ...
    'VariableNames', {'participant_id', 'group', 'condition', ...
    'mu2f_0', 'sa2f_0', 'mu2v_0', 'sa2v_0', ...
    'kappaF', 'thetaF', 'kappaV', 'thetaV', 'zeta', 'beta', 'aq_total'});

% Remove rows with missing data
valid = ~isnan(mu2f_0);
T = T(valid, :);
mu2f_0 = mu2f_0(valid); sa2f_0 = sa2f_0(valid);
mu2v_0 = mu2v_0(valid); sa2v_0 = sa2v_0(valid);
kappaF = kappaF(valid); thetaF = thetaF(valid);
kappaV = kappaV(valid); thetaV = thetaV(valid);
zeta = zeta(valid); beta = beta(valid);
group = group(valid); aq = aq(valid); condition = condition(valid);

outcsv = fullfile(resultdir, 'I1_Both_MAP_estimates.csv');
writetable(T, outcsv);
fprintf('Saved: %s\n\n', outcsv);

%% Group indices
isASD = strcmp(group, 'ASD');
isNT  = strcmp(group, 'NT');
nASD = sum(isASD);
nNT  = sum(isNT);
fprintf('Groups: ASD = %d, NT = %d\n\n', nASD, nNT);

%% 1. Descriptive statistics for initial conditions
fprintf('=== Initial Condition Parameters (I1_Both) ===\n');
fprintf('%-12s %10s %10s %10s %10s\n', 'Parameter', 'All_M', 'All_SD', 'ASD_M', 'NT_M');
fprintf('%s\n', repmat('-', 1, 55));

ic_params = {mu2f_0, sa2f_0, mu2v_0, sa2v_0};
ic_names  = {'mu2f_0', 'sa2f_0', 'mu2v_0', 'sa2v_0'};

for i = 1:4
    p = ic_params{i};
    fprintf('%-12s %10.4f %10.4f %10.4f %10.4f\n', ...
        ic_names{i}, mean(p), std(p), mean(p(isASD)), mean(p(isNT)));
end
fprintf('\n');

%% 2. T-tests: ASD vs NT on initial conditions
fprintf('=== T-tests: ASD vs NT (Initial Conditions) ===\n');
fprintf('%-12s %8s %8s %8s %8s %8s %8s\n', ...
    'Parameter', 'ASD_M', 'NT_M', 't', 'p', 'd', 'sig');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:4
    p = ic_params{i};
    asd_vals = p(isASD);
    nt_vals  = p(isNT);

    [~, pval, ~, stats] = ttest2(asd_vals, nt_vals);
    d_cohen = (mean(asd_vals) - mean(nt_vals)) / ...
        sqrt(((nASD-1)*var(asd_vals) + (nNT-1)*var(nt_vals)) / (nASD+nNT-2));

    if pval < 0.05, sig = '*'; else, sig = ''; end

    fprintf('%-12s %8.4f %8.4f %8.3f %8.4f %8.3f %8s\n', ...
        ic_names{i}, mean(asd_vals), mean(nt_vals), stats.tstat, pval, d_cohen, sig);
end
fprintf('\n');

%% 3. T-tests on standard parameters (from I1_Both model)
fprintf('=== T-tests: ASD vs NT (Standard Params, I1_Both) ===\n');
fprintf('%-12s %8s %8s %8s %8s %8s %8s\n', ...
    'Parameter', 'ASD_M', 'NT_M', 't', 'p', 'd', 'sig');
fprintf('%s\n', repmat('-', 1, 60));

std_params = {kappaF, thetaF, kappaV, thetaV, zeta, beta};
std_names  = {'kappaF', 'thetaF', 'kappaV', 'thetaV', 'zeta', 'beta'};

for i = 1:6
    p = std_params{i};
    asd_vals = p(isASD);
    nt_vals  = p(isNT);

    [~, pval, ~, stats] = ttest2(asd_vals, nt_vals);
    d_cohen = (mean(asd_vals) - mean(nt_vals)) / ...
        sqrt(((nASD-1)*var(asd_vals) + (nNT-1)*var(nt_vals)) / (nASD+nNT-2));

    if pval < 0.05, sig = '*'; else, sig = ''; end

    fprintf('%-12s %8.4f %8.4f %8.3f %8.4f %8.3f %8s\n', ...
        std_names{i}, mean(asd_vals), mean(nt_vals), stats.tstat, pval, d_cohen, sig);
end
fprintf('\n');

%% 4. AQ correlations with initial conditions
fprintf('=== Partial Correlations: AQ ~ Initial Conditions | Condition ===\n');

% Encode condition as numeric
condNum = NaN(length(condition), 1);
for i = 1:length(condition)
    cstr = char(condition{i});
    if isempty(cstr) || strcmp(cstr, '?')
        continue;
    end
    if contains(cstr, 'Facial', 'IgnoreCase', true) || ...
       contains(cstr, 'FF', 'IgnoreCase', true)
        condNum(i) = 0;
    elseif contains(cstr, 'Verbal', 'IgnoreCase', true) || ...
           contains(cstr, 'VF', 'IgnoreCase', true)
        condNum(i) = 1;
    end
end

valid_aq = ~isnan(aq) & ~isnan(condNum);
fprintf('Valid for AQ analysis: N = %d\n', sum(valid_aq));
fprintf('%-12s %10s %10s\n', 'Parameter', 'r_partial', 'p');
fprintf('%s\n', repmat('-', 1, 35));

for i = 1:4
    p = ic_params{i};
    if sum(valid_aq) > 5
        [r, pval] = partialcorr(aq(valid_aq), p(valid_aq), condNum(valid_aq));
        if pval < 0.05, sig = '*'; else, sig = ''; end
        fprintf('%-12s %10.4f %10.4f %s\n', ic_names{i}, r, pval, sig);
    end
end
fprintf('\n');

%% 5. Compare standard params: ArchA baseline vs I1_Both
fprintf('=== Parameter Stability: ArchA vs I1_Both ===\n');
fprintf('(Checking whether freeing initial conditions changes other parameter estimates)\n\n');

% Load baseline ArchA params from results_I1 (model 1)
baseline_prc = 'hgf_binary3l_facial_verbal_config';
kappaF_base = NaN(sum(valid), 1);
thetaF_base = NaN(sum(valid), 1);
kappaV_base = NaN(sum(valid), 1);
thetaV_base = NaN(sum(valid), 1);
zeta_base   = NaN(sum(valid), 1);
beta_base   = NaN(sum(valid), 1);

sids = T.participant_id;
for s = 1:length(sids)
    sid = sids{s};
    fname = fullfile(resultdir, sprintf('%s_%s_%s.mat', sid, baseline_prc, obs_config));
    if exist(fname, 'file')
        est_struct = load(fname);
        if isfield(est_struct, 'est') && isfield(est_struct.est, 'p_prc')
            kappaF_base(s) = est_struct.est.p_prc.ka_f;
            thetaF_base(s) = est_struct.est.p_prc.th_f;
            kappaV_base(s) = est_struct.est.p_prc.ka_v;
            thetaV_base(s) = est_struct.est.p_prc.th_v;
            zeta_base(s)   = est_struct.est.p_obs.ze1;
            beta_base(s)   = est_struct.est.p_obs.beta;
        end
    end
end

fprintf('%-12s %10s %10s %10s   %s\n', 'Parameter', 'r(base,I1)', 'mean_diff', 'paired_p', 'space');
fprintf('%s\n', repmat('-', 1, 60));

% Cross-model parameter stability is reported on the same scale as the rest
% of the report:
%   - kappa/theta : sigmoid-bounded native (no transform, est.p_prc.* already in [0,1])
%   - beta        : native (positive, est.p_obs.beta)
%   - zeta        : log-space (matches MLTM_load_zeta_new.m / MLTM_MAP_estimates.csv)
% The original implementation correlated zeta in native space because both
% sources expose est.p_obs.ze1 natively; we log-transform here so the
% stability r is directly comparable to the log-space zeta numbers used
% in the dimensional, group, sensitivity and stratified analyses.

base_params = {kappaF_base, thetaF_base, kappaV_base, thetaV_base, ...
               log(zeta_base), beta_base};
i1_params   = {kappaF, thetaF, kappaV, thetaV, ...
               log(zeta),      beta};
param_space = {'native', 'native', 'native', 'native', 'log', 'native'};

for i = 1:6
    bp = base_params{i};
    ip = i1_params{i};
    v = ~isnan(bp) & ~isnan(ip) & isfinite(bp) & isfinite(ip);
    if sum(v) > 3
        r = corr(bp(v), ip(v));
        [~, p_paired] = ttest(bp(v), ip(v));
        md = mean(ip(v) - bp(v));
        fprintf('%-12s %10.4f %10.4f %10.4f   %s\n', ...
            std_names{i}, r, md, p_paired, param_space{i});
    end
end

fprintf('\n=== I1 Analysis Complete ===\n');
fprintf('CSV saved: %s\n', outcsv);

end
