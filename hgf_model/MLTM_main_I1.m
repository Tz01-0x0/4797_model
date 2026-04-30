function MLTM_main_I1()
% Initial-condition variant analysis (separate from the main 4-model BMS).
% Fits three Arch-A variants that free the Level-2 priors:
%   I1-mean      : free mu2f_0, mu2v_0
%   I1-precision : free sa2f_0, sa2v_0
%   I1-both      : free mu2f_0, sa2f_0, mu2v_0, sa2v_0
% Then runs BMS over {ArchA baseline, I1-mean, I1-precision, I1-both}.
% Outputs go to hgf_model/results_I1/.

%% Setup
options = MLTM_options_new();

% Override: use I1 model space
options.model.perceptualModels = {
    'hgf_binary3l_facial_verbal_config', ...         % M1: ArchA baseline (for comparison)
    'hgf_binary3l_facial_verbal_I1mean_config', ...  % I1-mean: free mu2_0
    'hgf_binary3l_facial_verbal_I1precision_config', ... % I1-precision: free sa2_0
    'hgf_binary3l_facial_verbal_I1both_config', ...  % I1-both: free mu2_0 + sa2_0
};

options.model.responseModels = {
    'softmax_facial_verbal_config', ...
};

options.model.labels = {'M1_ArchA', 'I1_Mean', 'I1_Precision', 'I1_Both'};

% Override: save to separate results folder
options.resultroot = fullfile(options.code, 'results_I1');
if ~exist(options.resultroot, 'dir')
    mkdir(options.resultroot);
end

% Override: winning model for parameter extraction
% (For now, extract from baseline ArchA; will update after BMS)
options.model.winningPerceptual = 'hgf_binary3l_facial_verbal_config';
options.model.winningResponse   = 'softmax_facial_verbal_config';

% Override: extended parameter names for I1-both
% (For extraction, we still use the base 4+2 params from winning model)
options.model.hgfParam  = {'kappaF','thetaF','kappaV','thetaV'};
options.model.respParam = {'zeta','beta'};

%% Step 1: First-level inversion
fprintf('\n=== I1 Sub-Variant Analysis ===\n');
fprintf('Models: %s\n', strjoin(options.model.labels, ', '));
fprintf('Results: %s\n\n', options.resultroot);

fprintf('\nStep 1: First-level model inversion (I1 variants)\n');
MLTM_invert_subject_new(options);

%% Step 2: Model comparison only (BMS)
fprintf('\nStep 2: BMS across I1 variants\n');

% Only run model comparison, not full second-level
% (parameter extraction would use the winning I1 variant if it wins)
options.secondlevel = [1 0 0 0];  % [modelComparison only]
MLTM_second_level_new(options);

%% Step 3: Extract I1-specific parameters if needed
fprintf('\n=== I1 Analysis Complete ===\n');
fprintf('Check BMS results in: %s\n', options.resultroot);
fprintf('If an I1 variant wins, update MLTM_options and re-run parameter extraction.\n');

% Also extract and compare initial conditions across I1 models
extract_I1_parameters(options);

end


function extract_I1_parameters(options)
% Extract initial condition estimates from the BMS-winning I1 variant.

% Determines the winning model from BMS results, then extracts only
% that model's parameters. Falls back to I1_Both (index 4) if BMS
% results are unavailable.

fprintf('\n--- I1 Initial Condition Estimates ---\n');

% Determine winning model from BMS results
bms_file = fullfile(options.resultroot, 'BMS_results.mat');
winning_idx = 4;  % default: I1_Both
if exist(bms_file, 'file')
    bms = load(bms_file);
    if isfield(bms, 'xp')
        [~, winning_idx] = max(bms.xp);
    end
end
fprintf('Extracting from winning model: %s (index %d)\n', ...
    options.model.labels{winning_idx}, winning_idx);

% Load metadata for group assignment
meta = readtable(options.metadata);

prc_config = options.model.perceptualModels{winning_idx};
obs_config = options.model.responseModels{1};
nSubj = length(options.subjects);
results = struct('sid', cell(nSubj,1), 'group', cell(nSubj,1), ...
    'model', cell(nSubj,1), 'mu2f_0', cell(nSubj,1), ...
    'sa2f_0', cell(nSubj,1), 'mu2v_0', cell(nSubj,1), ...
    'sa2v_0', cell(nSubj,1));
nLoaded = 0;

for s = 1:nSubj
    sid = options.subjects{s};
    fname = fullfile(options.resultroot, ...
        sprintf('%s_%s_%s.mat', sid, prc_config, obs_config));

    if ~exist(fname, 'file')
        continue;
    end

    est = load(fname);
    if ~isfield(est, 'est') || ~isfield(est.est, 'p_prc')
        continue;
    end

    p = est.est.p_prc;
    nLoaded = nLoaded + 1;

    % Find group
    idx = find(strcmp(meta.participant_id, sid));
    if ~isempty(idx)
        group = meta.group{idx};
    else
        group = '?';
    end

    results(nLoaded).sid = sid;
    results(nLoaded).group = group;
    results(nLoaded).model = options.model.labels{winning_idx};
    results(nLoaded).mu2f_0 = p.mu2f_0;
    results(nLoaded).sa2f_0 = p.sa2f_0;
    results(nLoaded).mu2v_0 = p.mu2v_0;
    results(nLoaded).sa2v_0 = p.sa2v_0;
end

% Trim unused entries and save
if nLoaded > 0
    results = results(1:nLoaded);
    T = struct2table(results);
    outfile = fullfile(options.resultroot, 'I1_initial_conditions.csv');
    writetable(T, outfile);
    fprintf('Extracted %d subjects. Saved: %s\n', nLoaded, outfile);
else
    fprintf('WARNING: No subjects loaded.\n');
end

end
