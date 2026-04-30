function MLTM_sample_characterisation(data_dir)
% Build the publication Table 1 (descriptive statistics comparing ASD vs
% NT). Saves the rendered table as .png/.fig and prints the matching
% results paragraph. data_dir defaults to <repo_root>/data/.

if nargin < 1
    script_dir = fileparts(mfilename('fullpath'));
    if isempty(script_dir), script_dir = pwd; end
    data_dir = fullfile(script_dir, '..', 'data');
end

%% Load data
meta = readtable(fullfile(data_dir, 'participant_metadata.csv'), 'TextType', 'string');
qual = readtable(fullfile(data_dir, 'quality_report.csv'), 'TextType', 'string');

% Merge behavioral columns
df = outerjoin(meta, qual(:, {'participant_id','accuracy','mean_rt','median_rt', ...
    'follow_facial','follow_verbal','calibration_accuracy','timeout_rate'}), ...
    'Keys', 'participant_id', 'MergeKeys', true);

% Group assignment (reclassified: ASD vs NT only, no NT_highAQ)
is_asd = strcmp(df.group, 'ASD');
is_nt  = strcmp(df.group, 'NT');
asd = df(is_asd, :);
nt  = df(is_nt, :);
n_asd = height(asd);
n_nt  = height(nt);

fprintf('Sample: ASD n=%d, NT n=%d\n', n_asd, n_nt);

%% Define table rows
% Each row: {label, asd_string, nt_string, test_string}
rows = {};

% Gender
n_asd_m = sum(strcmp(asd.gender, 'Male'));
n_asd_f = sum(strcmp(asd.gender, 'Female'));
n_nt_m  = sum(strcmp(nt.gender, 'Male'));
n_nt_f  = sum(strcmp(nt.gender, 'Female'));
obs = [n_asd_m n_asd_f; n_nt_m n_nt_f];
[chi2_stat, ~, chi2_p] = chi2_test(obs);
rows{end+1} = {'Gender, M/F', sprintf('%d/%d', n_asd_m, n_asd_f), ...
    sprintf('%d/%d', n_nt_m, n_nt_f), format_chi2(chi2_stat, chi2_p)};

% Continuous variables
cont_vars = {
    'age',                   'Age, years'
    'aq_total',              'AQ-50'
    'aq_social_skill',       '  Social Skill'
    'aq_attention_switching', '  Attention Switching'
    'aq_attention_to_detail', '  Attention to Detail'
    'aq_communication',      '  Communication'
    'aq_imagination',        '  Imagination'
    'fi_score',              'Fluid Intelligence'
    'dass_total',            'DASS-21 Total'
    'dass_depression',       '  Depression'
    'dass_anxiety',          '  Anxiety'
    'dass_stress',           '  Stress'
};

% Compute DASS total
df.dass_total = df.dass_depression + df.dass_anxiety + df.dass_stress;
asd = df(is_asd, :);
nt  = df(is_nt, :);

for i = 1:size(cont_vars, 1)
    vname = cont_vars{i, 1};
    label = cont_vars{i, 2};
    x_asd = asd.(vname);
    x_nt  = nt.(vname);
    x_asd = x_asd(~isnan(x_asd));
    x_nt  = x_nt(~isnan(x_nt));

    [t, p, df_t] = welch_ttest(x_asd, x_nt);

    rows{end+1} = {label, ...
        format_mean_sem(x_asd), ...
        format_mean_sem(x_nt), ...
        format_t(t, p)};
end

%% Print console table
fprintf('\n');
fprintf('%-26s  %-16s  %-16s  %s\n', '', ...
    sprintf('ASD (n = %d)', n_asd), sprintf('NT (n = %d)', n_nt), '');
fprintf('%-26s  %-16s  %-16s  %s\n', ...
    'Variable', 'Mean (SD)', 'Mean (SD)', 'Test statistic');
fprintf('%s\n', repmat('-', 1, 80));

for r = 1:length(rows)
    fprintf('%-26s  %-16s  %-16s  %s\n', rows{r}{1}, rows{r}{2}, rows{r}{3}, rows{r}{4});
end
fprintf('%s\n', repmat('-', 1, 80));
fprintf('Values are mean (SD). Welch''s t-test for continuous; chi-squared for Gender.\n');
fprintf('* p < .05, ** p < .01\n\n');

%% Render as figure
fig = figure('Position', [80 80 820 520], 'Color', 'w', 'Name', 'Table 1');
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
hold on;

n_rows = length(rows);
row_h = 0.055;
y_top = 0.92;

% Colors
col_asd = [0.85 0.33 0.10];
col_nt  = [0.00 0.45 0.74];
col_sig = [0.75 0.10 0.10];

% Column x positions
cx = [0.02, 0.35, 0.55, 0.73];

% Title
text(0.5, 0.97, 'Table 1.  Descriptive Data of Participants', ...
    'FontSize', 13, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'Color', [0.4 0.15 0.05]);

% Top rule
line([0.01 0.99], [y_top+0.015 y_top+0.015], 'Color', [0 0 0.5], 'LineWidth', 2.5);

% Header
text(cx(1), y_top, 'Variable', 'FontSize', 10, 'FontWeight', 'bold');
text(cx(2), y_top, sprintf('ASD (n = %d)', n_asd), 'FontSize', 10, ...
    'FontWeight', 'bold', 'Color', col_asd);
text(cx(3), y_top, sprintf('NT (n = %d)', n_nt), 'FontSize', 10, ...
    'FontWeight', 'bold', 'Color', col_nt);
text(cx(4), y_top, 'Test', 'FontSize', 10, 'FontWeight', 'bold');

% Header rule
y_hr = y_top - 0.025;
line([0.01 0.99], [y_hr y_hr], 'Color', [0 0 0.5], 'LineWidth', 1.5);

% Data rows
for r = 1:n_rows
    y = y_hr - r * row_h;

    % Variable label
    label = rows{r}{1};
    if startsWith(label, '  ')
        % Subscale indent
        text(cx(1) + 0.02, y, label, 'FontSize', 9, 'Color', [0.3 0.3 0.3]);
    else
        text(cx(1), y, label, 'FontSize', 9.5, 'FontWeight', 'bold');
    end

    % ASD value
    text(cx(2), y, rows{r}{2}, 'FontSize', 9, 'Color', col_asd);

    % NT value
    text(cx(3), y, rows{r}{3}, 'FontSize', 9, 'Color', col_nt);

    % Test statistic — highlight if significant
    test_str = rows{r}{4};
    if contains(test_str, '*')
        text(cx(4), y, test_str, 'FontSize', 9, 'Color', col_sig, 'FontWeight', 'bold');
    else
        text(cx(4), y, test_str, 'FontSize', 9, 'Color', [0.3 0.3 0.3]);
    end
end

% Bottom rule
y_bot = y_hr - (n_rows + 0.5) * row_h;
line([0.01 0.99], [y_bot y_bot], 'Color', [0 0 0.5], 'LineWidth', 2.5);

% Footnote
text(0.02, y_bot - 0.03, ...
    'Values are reported as mean (SD). Welch''s t for continuous variables; \chi^2 for Gender.', ...
    'FontSize', 8, 'Color', [0.4 0.4 0.4]);

hold off;

% Save
if ~exist(fullfile(data_dir, 'figures'), 'dir')
    mkdir(fullfile(data_dir, 'figures'));
end
saveas(fig, fullfile(data_dir, 'figures', 'table1_sample.png'));
saveas(fig, fullfile(data_dir, 'figures', 'table1_sample.fig'));
fprintf('Saved: figures/table1_sample.png\n');

end

%% Helper functions

function s = format_mean_sem(x)
    m = mean(x);
    sd = std(x);
    if abs(m) >= 100
        s = sprintf('%.0f (%.0f)', m, sd);
    else
        s = sprintf('%.1f (%.1f)', m, sd);
    end
end

function s = format_t(t, p)
    star = '';
    if p < 0.001,     star = '***';
    elseif p < 0.01,  star = '**';
    elseif p < 0.05,  star = '*';
    end

    if p < 0.001
        s = sprintf('t = %.2f, p < .001 %s', t, star);
    else
        s = sprintf('t = %.2f, p = .%03d %s', t, round(p*1000), star);
    end
end

function s = format_chi2(chi2, p)
    star = '';
    if p < 0.01, star = '**'; elseif p < 0.05, star = '*'; end
    s = sprintf('\\chi^2 = %.2f, p = .%03d %s', chi2, round(p*1000), star);
end

function [t, p, df] = welch_ttest(x1, x2)
    n1 = length(x1); n2 = length(x2);
    m1 = mean(x1);   m2 = mean(x2);
    v1 = var(x1);    v2 = var(x2);
    t = (m1 - m2) / sqrt(v1/n1 + v2/n2);
    df = (v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1));
    x_val = df / (df + t^2);
    p = betainc(x_val, df/2, 0.5);
end

function [result, chi2, p] = chi2_test(observed)
    n = sum(observed(:));
    row_sums = sum(observed, 2);
    col_sums = sum(observed, 1);
    expected = row_sums * col_sums / n;
    chi2 = sum(sum((observed - expected).^2 ./ expected));
    p = 1 - gammainc(chi2/2, 0.5);
    result = chi2;
end
