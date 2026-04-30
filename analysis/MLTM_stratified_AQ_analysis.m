function MLTM_stratified_AQ_analysis()
% Supplementary discussion analyses (operates on per-subject MAP CSVs):
%   PART A: condition-stratified AQ zero-order correlations on ArchA and
%           I1_Both fits. Tests whether AQ ~ kappaF in Verbal-first and
%           AQ ~ zeta in Facial-first survive freeing initial conditions.
%           ArchA zeta is reported in both log and native space.
%   PART B: AQ ~ kappaF with task-quality covariates (calibration_accuracy,
%           follow_facial) within each subgroup.
% Inputs : hgf_model/results/MLTM_MAP_estimates.csv,
%          hgf_model/results_I1/I1_Both_MAP_estimates.csv,
%          data/quality_report.csv
% Output : hgf_model/results_stratified/
% Requires Statistics Toolbox (partialcorr).

%% Paths
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
repo_root   = fullfile(script_dir, '..');
archa_csv   = fullfile(repo_root, 'hgf_model', 'results',    'MLTM_MAP_estimates.csv');
i1both_csv  = fullfile(repo_root, 'hgf_model', 'results_I1', 'I1_Both_MAP_estimates.csv');
quality_csv = fullfile(repo_root, 'data', 'quality_report.csv');
outdir      = fullfile(repo_root, 'hgf_model', 'results_stratified');
if ~exist(outdir, 'dir'), mkdir(outdir); end

for f = {archa_csv, i1both_csv, quality_csv}
    if exist(f{1}, 'file') ~= 2, error('Missing: %s', f{1}); end
end

%% Load data
fprintf('Loading CSVs...\n');
T_archa = readtable(archa_csv);
T_i1    = readtable(i1both_csv);
T_qual  = readtable(quality_csv);
fprintf('  ArchA   : %d rows x %d cols\n', height(T_archa), width(T_archa));
fprintf('  I1_Both : %d rows x %d cols\n', height(T_i1),    width(T_i1));
fprintf('  Quality : %d rows x %d cols\n\n', height(T_qual), width(T_qual));

%% Normalise condition: FF/VF
T_qual.cond_std = normaliseCond(T_qual.condition);
T_i1.cond_std   = normaliseCond(T_i1.condition);

%% Merge condition + quality into ArchA / I1 (via participant_id)
T_archa.cond_std             = strings(height(T_archa), 1);
T_archa.calibration_accuracy = nan(height(T_archa), 1);
T_archa.follow_facial        = nan(height(T_archa), 1);
T_i1.calibration_accuracy    = nan(height(T_i1), 1);
T_i1.follow_facial           = nan(height(T_i1), 1);

qpid = string(T_qual.participant_id);

for i = 1:height(T_archa)
    k = find(qpid == string(T_archa.participant_id(i)), 1);
    if ~isempty(k)
        T_archa.cond_std(i)             = T_qual.cond_std(k);
        T_archa.calibration_accuracy(i) = T_qual.calibration_accuracy(k);
        T_archa.follow_facial(i)        = T_qual.follow_facial(k);
    end
end
for i = 1:height(T_i1)
    k = find(qpid == string(T_i1.participant_id(i)), 1);
    if ~isempty(k)
        T_i1.calibration_accuracy(i) = T_qual.calibration_accuracy(k);
        T_i1.follow_facial(i)        = T_qual.follow_facial(k);
    end
end

%% Sanity checks & warnings
fprintf('===== SANITY CHECKS =====\n');
fprintf('ArchA   : N=%d (FF=%d, VF=%d, missing=%d)\n', height(T_archa), ...
    sum(T_archa.cond_std=="FF"), sum(T_archa.cond_std=="VF"), sum(T_archa.cond_std==""));
fprintf('I1_Both : N=%d (FF=%d, VF=%d, missing=%d)\n', height(T_i1), ...
    sum(T_i1.cond_std=="FF"),    sum(T_i1.cond_std=="VF"),    sum(T_i1.cond_std==""));

% condition agreement ArchA vs I1_Both
[~, ia, ib] = intersect(string(T_archa.participant_id), string(T_i1.participant_id), 'stable');
n_match_cond = sum(T_archa.cond_std(ia) == T_i1.cond_std(ib));
fprintf('Condition ArchA==I1_Both : %d / %d\n', n_match_cond, numel(ia));

% kappaF agreement (sanity: should be r>.95)
r_sanity = corr(T_archa.kappaF(ia), T_i1.kappaF(ib), 'rows', 'complete');
fprintf('κf(ArchA) vs κf(I1_Both) : r = %.3f  (expect ≈ .97)\n', r_sanity);

% group label agreement ArchA vs I1_Both
g_archa = string(T_archa.group(ia));
g_i1    = string(T_i1.group(ib));
n_match_grp = sum(g_archa == g_i1);
if n_match_grp < numel(ia)
    warning('group label mismatch between ArchA and I1_Both: %d / %d', ...
        numel(ia) - n_match_grp, numel(ia));
end

% AQ agreement
aq_mismatch = sum(T_archa.aq_total(ia) ~= T_i1.aq_total(ib));
if aq_mismatch > 0
    warning('aq_total mismatch between ArchA and I1_Both: %d cases', aq_mismatch);
else
    fprintf('aq_total ArchA==I1_Both  : %d / %d\n', numel(ia), numel(ia));
end

% quality.group is known to have stale labels — flag but do not use.
fprintf('(NOTE: quality_report.group column is NOT used; ArchA/I1_Both groups are authoritative.)\n\n');

%% Add native-space zeta to ArchA (log -> native via exp)
T_archa.zeta_native = exp(T_archa.zeta);   % ArchA zeta is stored in log-space
% I1_Both zeta is already native-space; no transform needed.
fprintf('ArchA zeta range   (log):    [%.2f, %.2f]\n', min(T_archa.zeta),        max(T_archa.zeta));
fprintf('ArchA zeta range   (native): [%.2f, %.2f]\n', min(T_archa.zeta_native), max(T_archa.zeta_native));
fprintf('I1_Both zeta range (native): [%.2f, %.2f]\n\n', min(T_i1.zeta),          max(T_i1.zeta));

%% Part A: AQ ~ parameter, stratified by condition
fprintf('============================================================\n');
fprintf('  PART A.  AQ ~ parameter  [Pooled (ctrl Cond) | FF | VF]\n');
fprintf('============================================================\n');

% ArchA: report both zeta spaces
resA_archa = runStratifiedBlock('ArchA', T_archa, ...
    {'kappaF','thetaF','kappaV','thetaV','zeta','zeta_native','beta'});
% I1_Both: includes initial-condition parameters; zeta already native
resA_i1    = runStratifiedBlock('I1_Both', T_i1, ...
    {'kappaF','thetaF','kappaV','thetaV','zeta','beta','mu2f_0','sa2f_0','mu2v_0','sa2v_0'});

% Save Part A
writetable(resA_archa, fullfile(outdir, 'partA_ArchA_stratified.csv'));
writetable(resA_i1,    fullfile(outdir, 'partA_I1Both_stratified.csv'));

%% Part B: AQ ~ kappaF with task-quality covariates
fprintf('\n============================================================\n');
fprintf('  PART B.  AQ ~ kappaF | calibration_accuracy, follow_facial\n');
fprintf('============================================================\n');
fprintf('df for partial corr = n - 2 - k (k = #covariates)\n\n');
resB_archa = runCovariateBlock('ArchA',   T_archa);
resB_i1    = runCovariateBlock('I1_Both', T_i1);

writetable(resB_archa, fullfile(outdir, 'partB_ArchA_covariates.csv'));
writetable(resB_i1,    fullfile(outdir, 'partB_I1Both_covariates.csv'));

fprintf('\n=== Done. Outputs saved in: %s ===\n', outdir);
end  % main

% Helpers
function c = normaliseCond(raw)
% Accepts cellstr / string / char; returns 'FF' / 'VF' / ''.
    n = numel(raw); c = strings(n, 1);
    for ii = 1:n
        if iscell(raw), s = raw{ii}; else, s = raw(ii); end
        s = lower(string(s));
        if ismissing(s) || s == ""
            c(ii) = "";
        elseif contains(s, "facial") || contains(s, "ff")
            c(ii) = "FF";
        elseif contains(s, "verbal") || contains(s, "vf")
            c(ii) = "VF";
        end
    end
end

function res = runStratifiedBlock(label, T, paramNames)
% AQ vs each parameter: pooled (ctrl Cond), FF subgroup, VF subgroup.
% Returns a table of results.

fprintf('\n--- %s ---\n', label);
fprintf('%-12s  %-22s  %-22s  %-22s\n', 'Param', ...
    'Pooled (ctrl Cond)', 'Facial-first', 'Verbal-first');
fprintf('%s\n', repmat('-', 1, 84));

condNum = nan(height(T), 1);
condNum(T.cond_std == "FF") = 0;
condNum(T.cond_std == "VF") = 1;
aq = double(T.aq_total);

rows = {};
for ii = 1:numel(paramNames)
    pname = paramNames{ii};
    if ~ismember(pname, T.Properties.VariableNames)
        fprintf('%-12s  (not in table, skipped)\n', pname);
        continue;
    end
    y = double(T.(pname));

    v = ~isnan(aq) & ~isnan(y) & ~isnan(condNum);
    [r_p, p_p] = partialcorr(aq(v), y(v), condNum(v));
    np = sum(v);

    vFF = v & (T.cond_std == "FF");
    [r_ff, p_ff] = corr(aq(vFF), y(vFF));
    nff = sum(vFF);

    vVF = v & (T.cond_std == "VF");
    [r_vf, p_vf] = corr(aq(vVF), y(vVF));
    nvf = sum(vVF);

    % star if p<.05
    s_p  = sig(p_p);  s_ff = sig(p_ff);  s_vf = sig(p_vf);
    fprintf('%-12s  r=%+.3f p=%.3f N=%-2d%s  r=%+.3f p=%.3f N=%-2d%s  r=%+.3f p=%.3f N=%-2d%s\n', ...
        pname, r_p, p_p, np, s_p, r_ff, p_ff, nff, s_ff, r_vf, p_vf, nvf, s_vf);

    rows(end+1, :) = {pname, r_p, p_p, np, r_ff, p_ff, nff, r_vf, p_vf, nvf}; %#ok<AGROW>
end
res = cell2table(rows, 'VariableNames', ...
    {'parameter','r_pooled','p_pooled','N_pooled', ...
     'r_FF','p_FF','N_FF','r_VF','p_VF','N_VF'});
end

function res = runCovariateBlock(label, T)
fprintf('\n--- %s ---\n', label);
tags = {"FF","Facial-first"; "VF","Verbal-first"};
rows = {};
for sgi = 1:size(tags,1)
    tag  = tags{sgi,1}; name = tags{sgi,2};
    sub  = T(T.cond_std == tag, :);
    x  = double(sub.aq_total);
    y  = double(sub.kappaF);
    c1 = double(sub.calibration_accuracy);
    c2 = double(sub.follow_facial);
    v  = ~isnan(x) & ~isnan(y) & ~isnan(c1) & ~isnan(c2);
    n  = sum(v);
    fprintf('\n  %s  (N = %d)\n', name, n);
    if n < 6
        fprintf('    too few; skipping.\n'); continue;
    end
    fprintf('    calib  M=%.3f SD=%.3f  |  follow_facial  M=%.3f SD=%.3f\n', ...
        mean(c1(v)), std(c1(v)), mean(c2(v)), std(c2(v)));

    [r_ac,p_ac] = corr(x(v), c1(v));
    [r_af,p_af] = corr(x(v), c2(v));
    [r_kc,p_kc] = corr(y(v), c1(v));
    [r_kf,p_kf] = corr(y(v), c2(v));
    fprintf('    AQ ~ calib         : r=%+.3f p=%.3f%s\n', r_ac, p_ac, sig(p_ac));
    fprintf('    AQ ~ follow_facial : r=%+.3f p=%.3f%s\n', r_af, p_af, sig(p_af));
    fprintf('    kF ~ calib         : r=%+.3f p=%.3f%s\n', r_kc, p_kc, sig(p_kc));
    fprintf('    kF ~ follow_facial : r=%+.3f p=%.3f%s\n', r_kf, p_kf, sig(p_kf));

    [r0,p0]  = corr(x(v), y(v));
    [r1,p1]  = partialcorr(x(v), y(v), c1(v));
    [r2,p2]  = partialcorr(x(v), y(v), c2(v));
    [rb,pb]  = partialcorr(x(v), y(v), [c1(v) c2(v)]);
    fprintf('    AQ~kF  zero-order       : r=%+.3f p=%.4f df=%d%s\n', r0, p0, n-2, sig(p0));
    fprintf('    AQ~kF | calib           : r=%+.3f p=%.4f df=%d%s\n', r1, p1, n-3, sig(p1));
    fprintf('    AQ~kF | follow_facial   : r=%+.3f p=%.4f df=%d%s\n', r2, p2, n-3, sig(p2));
    fprintf('    AQ~kF | calib & fac     : r=%+.3f p=%.4f df=%d%s\n', rb, pb, n-4, sig(pb));

    rows(end+1, :) = {char(name), n, r0, p0, r1, p1, r2, p2, rb, pb}; %#ok<AGROW>
end
res = cell2table(rows, 'VariableNames', ...
    {'subgroup','N','r_zero','p_zero','r_calib','p_calib', ...
     'r_follow','p_follow','r_both','p_both'});
end

function s = sig(p)
    if     p < 0.001, s = ' ***';
    elseif p < 0.01,  s = ' **';
    elseif p < 0.05,  s = ' *';
    elseif p < 0.10,  s = ' .';
    else,             s = '';
    end
end
