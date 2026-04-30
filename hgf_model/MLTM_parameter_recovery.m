function MLTM_parameter_recovery()
% Simulation-recovery: simulate responses from each subject's fitted MAP
% parameters, refit, and compare true vs recovered parameters.
% Outputs: parameter_recovery_results.mat, _report.txt, _scatter.png.

%% Setup
options = MLTM_options_new();

prc_config = options.model.winningPerceptual;  % hgf_binary3l_facial_verbal_config
obs_config = options.model.winningResponse;    % softmax_facial_verbal_config
prc_fun_name = prc_config(1:end-7);  % remove _config
obs_fun_name = obs_config(1:end-7);

nSubj = length(options.subjects);

% Parameter names (matching the free parameters)
paramNames = {'kappaF', 'thetaF', 'kappaV', 'thetaV', 'zeta', 'beta'};
nParams = length(paramNames);

% Storage
true_params = NaN(nSubj, nParams);
recov_params = NaN(nSubj, nParams);
fit_success = false(nSubj, 1);

% Set random seed for reproducibility
rng(42, 'twister');

fprintf('\n========================================\n');
fprintf('  PARAMETER RECOVERY ANALYSIS\n');
fprintf('  Model: %s + %s\n', prc_config, obs_config);
fprintf('  Subjects: %d\n', nSubj);
fprintf('========================================\n\n');

%% Main loop
for s = 1:nSubj
    sid = options.subjects{s};
    fprintf('Subject %d/%d: %s ... ', s, nSubj, sid);

    %% 1. Load fitted model results
    matfile = fullfile(options.resultroot, ...
        sprintf('%s_%s_%s.mat', sid, prc_config, obs_config));

    if ~exist(matfile, 'file')
        fprintf('SKIP (no results file)\n');
        continue;
    end

    loaded = load(matfile);
    est = loaded.est;

    % Extract true MAP parameters (native space)
    true_params(s, 1) = est.p_prc.ka_f;
    true_params(s, 2) = est.p_prc.th_f;
    true_params(s, 3) = est.p_prc.ka_v;
    true_params(s, 4) = est.p_prc.th_v;
    true_params(s, 5) = est.p_obs.ze1;
    true_params(s, 6) = est.p_obs.beta;

    %% 2. Load original input sequence
    csv_path = fullfile(options.dataroot, [sid '_hgf_input.csv']);
    [~, inputs, ~] = MLTM_load_gorilla(csv_path);

    %% 3. Simulate responses using the generative model
    % Run perceptual model forward with true parameters
    prc_fun = str2func(prc_fun_name);

    % Build a minimal r structure for the perceptual model
    r_sim = struct();
    r_sim.u = inputs;
    r_sim.y = est.y;  % need original y for irr trials structure
    r_sim.c_prc = est.c_prc;
    r_sim.c_obs = est.c_obs;

    % Determine irregular trials (NaN responses)
    irr = [];
    for k = 1:size(r_sim.y, 1)
        if isnan(r_sim.y(k, 1))
            irr = [irr, k];
        end
    end
    % Also add ignored trials (NaN inputs)
    ign = [];
    for k = 1:size(r_sim.u, 1)
        if isnan(r_sim.u(k, 1))
            ign = [ign, k];
        end
    end
    r_sim.irr = unique([irr, ign]);
    r_sim.ign = ign;

    try
        % Run perceptual model to get trajectories
        [traj, infStates] = prc_fun(r_sim, est.p_prc.p);

        % Extract quantities needed for response simulation
        n = size(inputs, 1);

        % infStates dimensions: (n_trials, levels, trajectory_type)
        %   dim3: 1=muhat_f, 2=sahat_f, 3=muhat_v, 4=sahat_v
        % Must match softmax_facial_verbal.m indexing exactly.

        % Level 1 predictions
        x_f = infStates(:, 1, 1);      % muhat_f level 1
        x_v = infStates(:, 1, 3);      % muhat_v level 1

        % Level 3 predictions (volatility beliefs)
        mu3hat_f = infStates(:, 3, 1);  % muhat_f level 3
        mu3hat_v = infStates(:, 3, 3);  % muhat_v level 3

        % Cue directions
        facial_left = inputs(:, 3);
        verbal_left = inputs(:, 4);

        % Response model parameters
        ze1 = est.p_obs.ze1;
        beta_base = est.p_obs.beta;

        % Compute choice probabilities (same as softmax_facial_verbal.m)
        pf = 1 ./ (x_f .* (1 - x_f));
        pv = 1 ./ (x_v .* (1 - x_v));

        wf = ze1 .* pf ./ (ze1 .* pf + pv);
        wv = pv ./ (ze1 .* pf + pv);

        p_left_facial = facial_left .* x_f + (1 - facial_left) .* (1 - x_f);
        p_left_verbal = verbal_left .* x_v + (1 - verbal_left) .* (1 - x_v);

        b_left = wf .* p_left_facial + wv .* p_left_verbal;

        beta_eff = exp(-mu3hat_f - mu3hat_v + log(beta_base));

        v_diff = 2 .* b_left - 1;
        p_choose_left = 1 ./ (1 + exp(-beta_eff .* v_diff));

        % Clamp
        p_choose_left = max(min(p_choose_left, 1 - 1e-10), 1e-10);

        % Sample simulated responses
        sim_y = NaN(n, 1);
        for t = 1:n
            if ~ismember(t, r_sim.irr)
                sim_y(t) = double(rand() < p_choose_left(t));
            end
        end

        %% 4. Re-fit the model to simulated data
        est_recov = fitModel(sim_y, inputs, prc_config, obs_config);

        % Extract recovered parameters
        recov_params(s, 1) = est_recov.p_prc.ka_f;
        recov_params(s, 2) = est_recov.p_prc.th_f;
        recov_params(s, 3) = est_recov.p_prc.ka_v;
        recov_params(s, 4) = est_recov.p_prc.th_v;
        recov_params(s, 5) = est_recov.p_obs.ze1;
        recov_params(s, 6) = est_recov.p_obs.beta;

        fit_success(s) = true;
        fprintf('OK\n');

    catch ME
        fprintf('FAILED: %s\n', ME.message);
    end
end

%% 5. Compute recovery statistics
fprintf('\n========================================\n');
fprintf('  PARAMETER RECOVERY RESULTS\n');
fprintf('========================================\n\n');

valid = fit_success & ~any(isnan(true_params), 2) & ~any(isnan(recov_params), 2);
nValid = sum(valid);
fprintf('Successfully recovered: %d / %d subjects\n\n', nValid, nSubj);

% Open report file
reportFile = fullfile(options.resultroot, 'parameter_recovery_report.txt');
fid = fopen(reportFile, 'w');

printBoth(fid, '=== Parameter Recovery Results ===');
printBoth(fid, sprintf('Model: %s + %s', prc_config, obs_config));
printBoth(fid, sprintf('N valid: %d / %d', nValid, nSubj));
printBoth(fid, '');

printBoth(fid, sprintf('%-12s %8s %8s %8s %8s %8s', ...
    'Parameter', 'r', 'R^2', 'RMSE', 'bias', 'slope'));
printBoth(fid, repmat('-', 1, 60));

r_values = NaN(nParams, 1);
R2_values = NaN(nParams, 1);
rmse_values = NaN(nParams, 1);
bias_values = NaN(nParams, 1);
slope_values = NaN(nParams, 1);

for p = 1:nParams
    t = true_params(valid, p);
    r_p = recov_params(valid, p);

    % Pearson correlation
    r_val = corr(t, r_p);

    % R^2
    R2 = r_val^2;

    % RMSE
    rmse = sqrt(mean((t - r_p).^2));

    % Mean bias
    bias = mean(r_p - t);

    % Regression slope (should be ~1 for good recovery)
    b = [ones(length(t), 1), t] \ r_p;
    slope = b(2);

    r_values(p) = r_val;
    R2_values(p) = R2;
    rmse_values(p) = rmse;
    bias_values(p) = bias;
    slope_values(p) = slope;

    printBoth(fid, sprintf('%-12s %8.3f %8.3f %8.4f %8.4f %8.3f', ...
        paramNames{p}, r_val, R2, rmse, bias, slope));
end

printBoth(fid, '');
printBoth(fid, sprintf('Mean r across parameters: %.3f', nanmean(r_values)));
printBoth(fid, sprintf('Mean R^2 across parameters: %.3f', nanmean(R2_values)));

% Interpretation guide
printBoth(fid, '');
printBoth(fid, '--- Interpretation ---');
printBoth(fid, 'r > 0.8: Excellent recovery');
printBoth(fid, 'r 0.6-0.8: Good recovery');
printBoth(fid, 'r 0.4-0.6: Moderate recovery');
printBoth(fid, 'r < 0.4: Poor recovery');
printBoth(fid, 'slope ~1: Unbiased recovery');
printBoth(fid, 'slope < 1: Regression to mean (common)');

fclose(fid);
fprintf('\nReport saved: %s\n', reportFile);

%% 6. Save results
results = struct();
results.true_params = true_params;
results.recov_params = recov_params;
results.valid = valid;
results.paramNames = {paramNames};
results.r_values = r_values;
results.R2_values = R2_values;
results.rmse_values = rmse_values;
results.bias_values = bias_values;
results.slope_values = slope_values;
results.nValid = nValid;

save(fullfile(options.resultroot, 'parameter_recovery_results.mat'), 'results');
fprintf('Results saved: %s\n', fullfile(options.resultroot, 'parameter_recovery_results.mat'));

%% 7. Create scatter plots
try
    fig = figure('Position', [100 100 1200 800], 'Visible', 'off');

    for p = 1:nParams
        subplot(2, 3, p);
        t = true_params(valid, p);
        r_p = recov_params(valid, p);

        scatter(t, r_p, 30, 'filled', 'MarkerFaceAlpha', 0.6);
        hold on;

        % Identity line
        lims = [min([t; r_p]), max([t; r_p])];
        margin = 0.1 * diff(lims);
        lims = [lims(1) - margin, lims(2) + margin];
        plot(lims, lims, 'k--', 'LineWidth', 1);

        % Regression line
        b = [ones(length(t), 1), t] \ r_p;
        x_fit = linspace(lims(1), lims(2), 100);
        plot(x_fit, b(1) + b(2)*x_fit, 'r-', 'LineWidth', 1.5);

        xlabel('True');
        ylabel('Recovered');
        title(sprintf('%s (r=%.2f)', paramNames{p}, r_values(p)));
        xlim(lims);
        ylim(lims);
        axis square;
        grid on;
    end

    sgtitle('Parameter Recovery: True vs Recovered', 'FontSize', 14);

    saveas(fig, fullfile(options.resultroot, 'parameter_recovery_scatter.png'));
    fprintf('Figure saved: %s\n', fullfile(options.resultroot, 'parameter_recovery_scatter.png'));
    close(fig);
catch ME
    warning('Could not create figure: %s', ME.message);
end

fprintf('\n=== Parameter Recovery Complete ===\n');

end

%% Helper function
function printBoth(fid, str)
    fprintf('%s\n', str);
    fprintf(fid, '%s\n', str);
end
