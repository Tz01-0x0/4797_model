function MLTM_laplace_kappa_theta(options)
% Per-subject Laplace posterior correlation between kappa and theta from
% est.optim.Corr (= -H^{-1} at MAP). Distinguishes a flat likelihood
% (uniformly small |r|, posterior pulled to prior) from a kappa<->theta
% ridge (large |r|, true trade-off). Outputs text + CSV + histogram.
% Parameter index order: 1=kF, 2=thF, 3=kV, 4=thV, 5=zeta(log), 6=beta(log).

if nargin < 1
    options = MLTM_options_new();
end

prc_config = options.model.winningPerceptual;
obs_config = options.model.winningResponse;

% Free parameter ordering for ArchA (6 free params)
idx_kf = 1;
idx_tf = 2;
idx_kv = 3;
idx_tv = 4;

subjAll = options.subjects;
nSubj   = numel(subjAll);

pid     = cell(nSubj, 1);
r_kf_tf = NaN(nSubj, 1);
r_kv_tv = NaN(nSubj, 1);

nLoaded = 0;

fprintf('\n=== Laplace within-subject κ–θ posterior correlation ===\n');
fprintf('Reading est.optim.Corr from %d ArchA fits...\n', nSubj);

for i = 1:nSubj
    sid = subjAll{i};
    fname = fullfile(options.resultroot, ...
        [sid, '_', prc_config, '_', obs_config, '.mat']);
    if exist(fname, 'file') ~= 2
        warning('Missing fit for %s, skipped.', sid);
        continue;
    end
    tmp = load(fname, 'est');
    if ~isfield(tmp, 'est') || ~isfield(tmp.est, 'optim') || ...
            ~isfield(tmp.est.optim, 'Corr')
        warning('No optim.Corr for %s, skipped.', sid);
        continue;
    end

    C = tmp.est.optim.Corr;
    if size(C,1) < 4 || size(C,2) < 4
        warning('Corr too small for %s (%dx%d), skipped.', ...
            sid, size(C,1), size(C,2));
        continue;
    end

    nLoaded         = nLoaded + 1;
    pid{nLoaded}    = sid;
    r_kf_tf(nLoaded) = C(idx_kf, idx_tf);
    r_kv_tv(nLoaded) = C(idx_kv, idx_tv);
end

% Trim
pid     = pid(1:nLoaded);
r_kf_tf = r_kf_tf(1:nLoaded);
r_kv_tv = r_kv_tv(1:nLoaded);

% Aggregate stats
m_kf  = mean(r_kf_tf);     m_kv  = mean(r_kv_tv);
md_kf = median(r_kf_tf);   md_kv = median(r_kv_tv);
mx_kf = max(abs(r_kf_tf)); mx_kv = max(abs(r_kv_tv));
n5_kf = sum(abs(r_kf_tf) >= 0.5);
n5_kv = sum(abs(r_kv_tv) >= 0.5);

% Print report
report = sprintf([ ...
    '=== Laplace posterior κ–θ correlation (ArchA, N=%d) ===\n' ...
    '\n' ...
    'Source: est.optim.Corr from each subject''s ArchA fit\n' ...
    '        (computed as Cov2Corr(inv(H_negLogJoint at MAP)) in fitModel.m)\n' ...
    '\n' ...
    '%-8s   %8s   %8s   %8s   %8s   %14s\n' ...
    '%s\n' ...
    '%-8s   %+8.3f   %+8.3f   %+8.3f   %+8.3f   %14d\n' ...
    '%-8s   %+8.3f   %+8.3f   %+8.3f   %+8.3f   %14d\n' ...
    '\n' ...
    'Interpretation guide:\n' ...
    '  |r| uniformly small (≤ ~.5) ⇒ flat likelihood, MAP shrunk to prior\n' ...
    '                                ⇒ observed κ/θ effects are conservative\n' ...
    '  |r| close to 1 across subjects ⇒ ridged likelihood (κ↔θ trade-off)\n' ...
    '                                  ⇒ effects could be artefactual\n'], ...
    nLoaded, ...
    'Pair', 'mean r', 'median', 'max |r|', 'min r', 'N(|r| >= .5)', ...
    repmat('-', 1, 70), ...
    'kf-tf', m_kf, md_kf, mx_kf, min(r_kf_tf), n5_kf, ...
    'kv-tv', m_kv, md_kv, mx_kv, min(r_kv_tv), n5_kv);

fprintf('\n%s', report);

% Save .txt
txtfile = fullfile(options.resultroot, 'laplace_kappa_theta.txt');
fid = fopen(txtfile, 'w');
fprintf(fid, '%s', report);

% Per-subject listing for transparency
fprintf(fid, '\n--- Per-subject |r| values ---\n');
fprintf(fid, '%-30s   %10s   %10s\n', 'participant_id', 'r_kf_thf', 'r_kv_thv');
for i = 1:nLoaded
    fprintf(fid, '%-30s   %+10.3f   %+10.3f\n', pid{i}, r_kf_tf(i), r_kv_tv(i));
end
fclose(fid);
fprintf('Saved text report: %s\n', txtfile);

% Save .csv
csvfile = fullfile(options.resultroot, 'laplace_kappa_theta.csv');
T = table(pid, r_kf_tf, r_kv_tv, ...
    'VariableNames', {'participant_id', 'r_kf_thf', 'r_kv_thv'});
writetable(T, csvfile);
fprintf('Saved CSV: %s\n', csvfile);

% Histogram figure
try
    fig = figure('Visible', 'off', 'Position', [100 100 900 400]);

    subplot(1,2,1);
    histogram(r_kf_tf, -1:0.1:1, 'FaceColor', [.3 .5 .8]);
    hold on;
    line([0.5 0.5],   ylim, 'Color', 'r', 'LineStyle', '--');
    line([-0.5 -0.5], ylim, 'Color', 'r', 'LineStyle', '--');
    title(sprintf('\\kappa_f - \\theta_f  (mean=%.3f, max|r|=%.3f)', ...
        m_kf, mx_kf));
    xlabel('Laplace posterior r'); ylabel('# subjects');
    xlim([-1 1]); grid on;

    subplot(1,2,2);
    histogram(r_kv_tv, -1:0.1:1, 'FaceColor', [.8 .4 .3]);
    hold on;
    line([0.5 0.5],   ylim, 'Color', 'r', 'LineStyle', '--');
    line([-0.5 -0.5], ylim, 'Color', 'r', 'LineStyle', '--');
    title(sprintf('\\kappa_v - \\theta_v  (mean=%.3f, max|r|=%.3f)', ...
        m_kv, mx_kv));
    xlabel('Laplace posterior r'); ylabel('# subjects');
    xlim([-1 1]); grid on;

    sgtitle(sprintf('Within-subject Laplace posterior correlation  (N=%d)', nLoaded));
    pngfile = fullfile(options.resultroot, 'laplace_kappa_theta_hist.png');
    saveas(fig, pngfile);
    close(fig);
    fprintf('Saved histogram: %s\n', pngfile);
catch ME
    warning('Could not create figure: %s', ME.message);
end

fprintf('\nDone.\n');

end
