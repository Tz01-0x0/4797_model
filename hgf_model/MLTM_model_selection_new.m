function MLTM_model_selection_new(options)
% Random-effects Bayesian Model Selection (Stephan et al., 2009) over all
% model combinations in options.model.labels. Saves BMS results and the
% per-subject log-Bayes-factor / posterior / exceedance-probability plots.
% Requires SPM12 on path (spm_BMS).
%
% NOTE on numerical reproducibility:
%   spm_BMS estimates exceedance probabilities (xp) by Gibbs sampling from
%   the Dirichlet posterior, with no fixed random seed. Re-running this
%   script therefore yields slightly different xp values from one run to
%   the next, typically within <=1 percentage point of the manuscript values
%   (e.g., ArchA xp has been observed at .966 on one run and .973 on
%   another). The posterior expectations <r_m> are analytic and stable, and
%   the model ordering does not change. See MODEL.md section 4.2.

%% Extract log-model-evidence
[models] = MLTM_extractLME_new(options);

save(fullfile(options.resultroot, 'models_F_values.mat'), 'models', '-mat');

nModels  = size(models, 2);
nSubjects = size(models, 1);

%% Bayesian Model Selection
if ~exist('spm_BMS', 'file')
    error(['spm_BMS not found. Add SPM12 to your path.\n' ...
           'Set options.spmpath in MLTM_options_new.m']);
end

[~, model_posterior, xp, protected_xp, ~] = spm_BMS(models);

% Save BMS results
bms_results.model_posterior = model_posterior;
bms_results.exceedance_prob = xp;
bms_results.protected_xp   = protected_xp;
bms_results.F_matrix        = models;
bms_results.model_labels    = options.model.labels;
save(fullfile(options.resultroot, 'BMS_results.mat'), 'bms_results', '-mat');

%% Identify best model
[~, best_model]  = max(model_posterior);
[~, worst_model] = min(model_posterior);

fprintf('\n=== Bayesian Model Selection Results ===\n');
for m = 1:nModels
    fprintf('  %s: posterior = %.4f, xp = %.4f\n', ...
        options.model.labels{m}, model_posterior(m), xp(m));
end
fprintf('  Best model: %s (index %d)\n', options.model.labels{best_model}, best_model);

%% Figure 1: Per-subject log Bayes factor relative to worst model
figure('Name', 'Log Bayes Factors');
log_bfs = models - repmat(models(:, worst_model), [1, nModels]);
handleBar = bar(log_bfs(:, best_model));
set(handleBar, 'FaceColor', [0.3 0.3 0.3]);
hold on;
line([0 nSubjects+1], [3 3], 'LineWidth', 2, 'LineStyle', '-.', 'Color', [1 0 0]);
line([0 nSubjects+1], [-3 -3], 'LineWidth', 2, 'LineStyle', '-.', 'Color', [1 0 0]);
ylabel('Log Bayes Factor');
xlabel('Subject');
title(sprintf('Log BF: %s vs %s', options.model.labels{best_model}, options.model.labels{worst_model}));
hold off;

%% Figure 2: Model posterior probabilities
figure('Name', 'Model Posterior');
colors = lines(nModels);
for i = 1:nModels
    h = bar(i, model_posterior(i));
    if i == 1, hold on; end
    set(h, 'FaceColor', colors(i,:));
end
set(gca, 'XTick', 1:nModels);
set(gca, 'XTickLabel', options.model.labels);
ylabel('p(r|y)');
title('Model Posterior Probabilities');
hold off;

%% Figure 3: Exceedance probabilities
figure('Name', 'Exceedance Probabilities');
for i = 1:nModels
    j = bar(i, xp(i));
    if i == 1, hold on; end
    set(j, 'FaceColor', colors(i,:));
end
set(gca, 'XTick', 1:nModels);
set(gca, 'XTickLabel', options.model.labels);
ylabel('Exceedance Probability');
title('Exceedance Probabilities');
hold off;

%% Save figures
saveas(gcf, fullfile(options.resultroot, 'BMS_exceedance.png'));
fprintf('BMS figures saved to %s\n', options.resultroot);

end
