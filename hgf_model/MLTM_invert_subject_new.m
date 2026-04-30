function MLTM_invert_subject_new(options)
% First-level inversion: fit every (perceptual x response) model combination
% to every subject in options.subjects via variational Bayes (fitModel).
% Output: <resultroot>/<subjID>_<prc_config>_<obs_config>.mat
%
% Set options.invert_overwrite = false to skip subject x model combinations
% whose .mat already exists on disk (default: true, i.e. always re-fit).

subjAll = options.subjects;
nPerc   = length(options.model.perceptualModels);
nResp   = length(options.model.responseModels);

% Default: re-fit everything. Set false to reuse existing fits as a cache.
if isfield(options, 'invert_overwrite')
    overwrite = logical(options.invert_overwrite);
else
    overwrite = true;
end

% Build all perceptual × response model combinations
nModels = nPerc * nResp;
iCombPercResp = zeros(nModels, 2);
idx = 0;
for iResp = 1:nResp
    for iPerc = 1:nPerc
        idx = idx + 1;
        iCombPercResp(idx, 1) = iPerc;
        iCombPercResp(idx, 2) = iResp;
    end
end

fprintf('Running %d model(s) on %d subject(s)...\n', nModels, length(subjAll));

for i = 1:length(subjAll)
    subjID = subjAll{i};
    fprintf('\n=== Subject %s (%d/%d) ===\n', subjID, i, length(subjAll));

    % Load data using Gorilla loader
    csv_path = fullfile(options.dataroot, [subjID '_hgf_input.csv']);
    if ~exist(csv_path, 'file')
        warning('Data file not found: %s. Skipping.', csv_path);
        continue;
    end

    [responses, inputs, metadata] = MLTM_load_gorilla(csv_path);

    for iModel = 1:nModels
        prc_config = options.model.perceptualModels{iCombPercResp(iModel,1)};
        obs_config = options.model.responseModels{iCombPercResp(iModel,2)};

        save_name = [subjID, '_', prc_config, '_', obs_config, '.mat'];
        save_path = fullfile(options.resultroot, save_name);

        if ~overwrite && exist(save_path, 'file') == 2
            fprintf('  Model %d/%d: %s + %s  [cached, skipping]\n', ...
                    iModel, nModels, prc_config, obs_config);
            continue;
        end

        fprintf('  Model %d/%d: %s + %s\n', iModel, nModels, prc_config, obs_config);

        try
            est = fitModel(responses, inputs, prc_config, obs_config);

            % Save results
            save(save_path, 'est', 'metadata', '-mat');
            fprintf('  -> Saved: %s\n', save_name);
            fprintf('  -> F = %.2f\n', est.F);

        catch ME
            warning('  Model fitting failed for %s, model %d: %s', subjID, iModel, ME.message);
        end
    end
end

fprintf('\nDone. Results saved to %s\n', options.resultroot);

end
