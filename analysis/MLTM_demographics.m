%% MLTM_demographics.m
% Demographic equivalence and behavioural group comparisons (ASD vs NT):
% age, gender, FI, AQ50 (total + subscales), DASS21, task accuracy,
% DASS sensitivity analysis. Console output only (no figures).
% Statistics Toolbox is used if present, otherwise manual fallbacks.
% Reads participant_metadata.csv and quality_report.csv from data/.

clear; close all; clc;

%% Setup
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
data_root  = fullfile(script_dir, '..', 'data');

% Check for Statistics Toolbox
has_stats_toolbox = ~isempty(ver('stats'));
if has_stats_toolbox
    fprintf('Statistics Toolbox detected.\n');
else
    fprintf('Statistics Toolbox NOT found. Using manual implementations.\n');
end

%% Load Data
meta = readtable(fullfile(data_root, 'participant_metadata.csv'), ...
    'TextType', 'string');
qr = readtable(fullfile(data_root, 'quality_report.csv'), ...
    'TextType', 'string');

% Pull only the behavioural columns from quality_report. The shipped qr
% does duplicate the canonical group / aq_total fields (authoritative in
% participant_metadata.csv), so this select removes them to avoid a name
% collision in outerjoin.
qr_behav = qr(:, {'participant_id', 'accuracy', 'mean_rt', 'median_rt', ...
                  'follow_facial', 'follow_verbal', 'calibration_accuracy', ...
                  'timeout_rate', 'condition'});

% Merge metadata with behavioural quality report
data = outerjoin(meta, qr_behav, 'Keys', 'participant_id', 'MergeKeys', true);

% Clean age to numeric
if iscell(data.age) || isstring(data.age)
    data.age_numeric = str2double(data.age);
else
    data.age_numeric = data.age;
end

% Exclude 100% timeout participant from behavioral analyses
data_behav = data(data.timeout_rate < 1.0, :);

% Group indices
idx_ASD = strcmp(data.group, 'ASD');
idx_NT  = strcmp(data.group, 'NT');
idx_ASD_b = strcmp(data_behav.group, 'ASD');
idx_NT_b  = strcmp(data_behav.group, 'NT');

% Counter for any rows that survived as the (now-deprecated) NT_highAQ
% category — kept only as a safety check; canonical labels never produce one.
idx_haq3  = strcmp(data.group, 'NT_highAQ');

fprintf('\n========================================\n');
fprintf('SAMPLE OVERVIEW\n');
fprintf('========================================\n');
fprintf('Total: %d participants\n', height(data));
fprintf('  ASD (formal diagnosis): %d\n', sum(idx_ASD));
fprintf('  NT (no diagnosis):      %d\n', sum(idx_NT));
fprintf('  -- of which NT_highAQ:  %d\n', sum(idx_haq3));
fprintf('  Behavioral sample:      %d (excl 100%% timeout)\n', height(data_behav));


%% Statistical test functions

function [t_stat, p_val, df] = welch_ttest(x, y)
    % Welch's t-test (unequal variance)
    nx = length(x); ny = length(y);
    mx = mean(x);   my = mean(y);
    vx = var(x);    vy = var(y);

    t_stat = (mx - my) / sqrt(vx/nx + vy/ny);

    % Welch-Satterthwaite degrees of freedom
    num = (vx/nx + vy/ny)^2;
    den = (vx/nx)^2/(nx-1) + (vy/ny)^2/(ny-1);
    df = num / den;

    % Two-tailed p-value using t-distribution
    % If no stats toolbox, use normal approximation for large df
    if ~isempty(ver('stats'))
        p_val = 2 * tcdf(-abs(t_stat), df);
    else
        % Normal approximation (good for df > 30)
        if df > 30
            p_val = 2 * (1 - normcdf_manual(abs(t_stat)));
        else
            p_val = 2 * tcdf_manual(-abs(t_stat), df);
        end
    end
end

function p = normcdf_manual(x)
    % Manual standard normal CDF using error function
    p = 0.5 * (1 + erf(x / sqrt(2)));
end

function p = tcdf_manual(t, df)
    % Approximate t-CDF using normal for large df,
    % or numerical integration for small df
    if df > 100
        p = normcdf_manual(t);
    else
        % Use betainc regularized incomplete beta function
        x = df / (df + t^2);
        p = 0.5 * betainc(x, df/2, 0.5);
        if t > 0
            p = 1 - p;
        end
    end
end

function [U, p_val] = mannwhitney(x, y)
    % Mann-Whitney U test
    nx = length(x); ny = length(y);
    combined = [x(:); y(:)];
    [~, idx] = sort(combined);
    ranks = zeros(size(combined));
    ranks(idx) = 1:length(combined);

    % Handle ties: average ranks
    [sorted_vals, sort_idx] = sort(combined);
    avg_ranks = ranks;
    i = 1;
    while i <= length(sorted_vals)
        j = i;
        while j <= length(sorted_vals) && sorted_vals(j) == sorted_vals(i)
            j = j + 1;
        end
        if j > i + 1  % ties exist
            tie_rank = mean(i:j-1);
            avg_ranks(sort_idx(i:j-1)) = tie_rank;
        end
        i = j;
    end

    R1 = sum(avg_ranks(1:nx));
    U1 = R1 - nx*(nx+1)/2;
    U2 = nx*ny - U1;
    U = min(U1, U2);

    % Normal approximation for p-value
    mu_U = nx*ny/2;
    sigma_U = sqrt(nx*ny*(nx+ny+1)/12);
    z = (U - mu_U) / sigma_U;
    p_val = 2 * (1 - normcdf_manual(abs(z)));
end

function d = cohens_d(x, y)
    % Cohen's d with pooled SD
    nx = length(x); ny = length(y);
    pooled_sd = sqrt(((nx-1)*var(x) + (ny-1)*var(y)) / (nx+ny-2));
    if pooled_sd > 0
        d = (mean(x) - mean(y)) / pooled_sd;
    else
        d = 0;
    end
end

function sig_str = sig_stars(p)
    if p < 0.001
        sig_str = '***';
    elseif p < 0.01
        sig_str = '**';
    elseif p < 0.05
        sig_str = '*';
    else
        sig_str = 'ns';
    end
end

function result = compare_groups(x_asd, x_nt, var_name)
    % Run full comparison suite for a continuous variable
    x_asd = x_asd(~isnan(x_asd));
    x_nt  = x_nt(~isnan(x_nt));

    result.variable = var_name;
    result.asd_m = mean(x_asd);  result.asd_sd = std(x_asd);  result.asd_n = length(x_asd);
    result.nt_m  = mean(x_nt);   result.nt_sd  = std(x_nt);   result.nt_n  = length(x_nt);

    [result.t, result.p_t, result.df] = welch_ttest(x_asd, x_nt);
    [result.U, result.p_U] = mannwhitney(x_asd, x_nt);
    result.d = cohens_d(x_asd, x_nt);
    result.sig = sig_stars(min(result.p_t, result.p_U));

    fprintf('  %-25s ASD M=%.2f(SD=%.2f), NT M=%.2f(SD=%.2f)\n', ...
        var_name, result.asd_m, result.asd_sd, result.nt_m, result.nt_sd);
    fprintf('  %25s t=%.2f, p=%.3f | U p=%.3f | d=%.2f [%s]\n', ...
        '', result.t, result.p_t, result.p_U, result.d, result.sig);
end


%% 1. Demographic equivalence

fprintf('\n========================================\n');
fprintf('1. DEMOGRAPHIC EQUIVALENCE (ASD vs NT)\n');
fprintf('   Goal: no significant differences\n');
fprintf('========================================\n\n');

results = {};

% Age
r = compare_groups(data.age_numeric(idx_ASD), data.age_numeric(idx_NT), 'Age');
results{end+1} = r;

% Fluid Intelligence
r = compare_groups(data.fi_score(idx_ASD), data.fi_score(idx_NT), 'Fluid Intelligence');
results{end+1} = r;

% Gender - Chi-square
fprintf('\n  Gender:\n');
gender_asd = data.gender(idx_ASD);
gender_nt  = data.gender(idx_NT);
n_female_asd = sum(strcmp(gender_asd, 'Female'));
n_male_asd   = sum(strcmp(gender_asd, 'Male'));
n_female_nt  = sum(strcmp(gender_nt, 'Female'));
n_male_nt    = sum(strcmp(gender_nt, 'Male'));
fprintf('    ASD: %d Female, %d Male\n', n_female_asd, n_male_asd);
fprintf('    NT:  %d Female, %d Male\n', n_female_nt, n_male_nt);

% Chi-square test for gender
observed = [n_female_asd, n_male_asd; n_female_nt, n_male_nt];
row_totals = sum(observed, 2);
col_totals = sum(observed, 1);
grand_total = sum(observed(:));
expected = row_totals * col_totals / grand_total;
chi2 = sum(sum((observed - expected).^2 ./ expected));
p_chi2 = 1 - normcdf_manual(sqrt(chi2));  % approximate for 1 df
if has_stats_toolbox
    p_chi2 = 1 - chi2cdf(chi2, 1);
end
fprintf('    Chi2 = %.2f, p = %.3f [%s]\n', chi2, p_chi2, sig_stars(p_chi2));


%% 2. Clinical measures

fprintf('\n========================================\n');
fprintf('2. CLINICAL MEASURES (ASD vs NT)\n');
fprintf('========================================\n\n');

clinical_vars = {
    'aq_total',               'AQ Total';
    'aq_social_skill',        'AQ Social Skill';
    'aq_attention_switching',  'AQ Attention Switching';
    'aq_attention_to_detail',  'AQ Attention to Detail';
    'aq_communication',        'AQ Communication';
    'aq_imagination',          'AQ Imagination';
    'dass_depression',         'DASS Depression';
    'dass_anxiety',            'DASS Anxiety';
    'dass_stress',             'DASS Stress';
};

for i = 1:size(clinical_vars, 1)
    var = clinical_vars{i, 1};
    label = clinical_vars{i, 2};
    r = compare_groups(data.(var)(idx_ASD), data.(var)(idx_NT), label);
    results{end+1} = r;
end


%% 3. Task behavioural performance

fprintf('\n========================================\n');
fprintf('3. TASK PERFORMANCE (ASD vs NT)\n');
fprintf('========================================\n\n');

task_vars = {
    'accuracy',               'Accuracy';
    'mean_rt',                'Mean RT (ms)';
    'follow_facial',          'Follow Facial Cue';
    'follow_verbal',          'Follow Verbal Cue';
    'calibration_accuracy',   'Calibration Accuracy';
    'timeout_rate',           'Timeout Rate';
};

for i = 1:size(task_vars, 1)
    var = task_vars{i, 1};
    label = task_vars{i, 2};
    r = compare_groups(data_behav.(var)(idx_ASD_b), data_behav.(var)(idx_NT_b), label);
    results{end+1} = r;
end


%% 4. DASS sensitivity analysis

fprintf('\n========================================\n');
fprintf('4. DASS SENSITIVITY ANALYSIS\n');
fprintf('========================================\n');

% Clinical DASS: Severe or Extremely Severe on any subscale
dass_clinical = (data_behav.dass_depression >= 21) | ...
                (data_behav.dass_anxiety >= 15) | ...
                (data_behav.dass_stress >= 26);

fprintf('\n  Severe+ DASS: %d / %d participants\n', sum(dass_clinical), height(data_behav));
fprintf('    ASD: %d, NT: %d\n', ...
    sum(dass_clinical & idx_ASD_b), sum(dass_clinical & idx_NT_b));

data_nodass = data_behav(~dass_clinical, :);
idx_ASD_nd = strcmp(data_nodass.group, 'ASD');
idx_NT_nd  = strcmp(data_nodass.group, 'NT');

fprintf('\n  After exclusion: %d remaining (ASD=%d, NT=%d)\n', ...
    height(data_nodass), sum(idx_ASD_nd), sum(idx_NT_nd));

fprintf('\n  --- WITH vs WITHOUT DASS exclusion ---\n\n');
sens_vars = {'accuracy', 'mean_rt', 'follow_facial', 'follow_verbal'};
sens_labels = {'Accuracy', 'Mean RT', 'Follow Facial', 'Follow Verbal'};

for i = 1:length(sens_vars)
    var = sens_vars{i};

    [t1, p1, ~] = welch_ttest(data_behav.(var)(idx_ASD_b), data_behav.(var)(idx_NT_b));
    d1 = cohens_d(data_behav.(var)(idx_ASD_b), data_behav.(var)(idx_NT_b));

    [t2, p2, ~] = welch_ttest(data_nodass.(var)(idx_ASD_nd), data_nodass.(var)(idx_NT_nd));
    d2 = cohens_d(data_nodass.(var)(idx_ASD_nd), data_nodass.(var)(idx_NT_nd));

    fprintf('  %s:\n', sens_labels{i});
    fprintf('    Full:      t=%.2f, p=%.3f, d=%.2f [%s]\n', t1, p1, d1, sig_stars(p1));
    fprintf('    Excl DASS: t=%.2f, p=%.3f, d=%.2f [%s]\n\n', t2, p2, d2, sig_stars(p2));
end


fprintf('\nDone.\n');
