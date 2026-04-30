function [models] = MLTM_extractLME_new(options)
% Build the [nSubjects x nModels] matrix of log-model-evidence (variational
% free energy F) used as input to spm_BMS.

subjAll = options.subjects;
nPerc   = length(options.model.perceptualModels);
nResp   = length(options.model.responseModels);

% Build all perceptual x response combinations (same order as inversion)
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

models = NaN(numel(subjAll), nModels);

for iSubject = 1:numel(subjAll)
    for iModel = 1:nModels
        prc_config = options.model.perceptualModels{iCombPercResp(iModel,1)};
        obs_config = options.model.responseModels{iCombPercResp(iModel,2)};

        fname = fullfile(options.resultroot, ...
            [subjAll{iSubject}, '_', prc_config, '_', obs_config, '.mat']);

        if ~exist(fname, 'file')
            warning('Result file not found: %s', fname);
            continue;
        end

        tmp = load(fname, 'est');
        models(iSubject, iModel) = tmp.est.F;
    end
end

% Report any missing values
nMissing = sum(isnan(models(:)));
if nMissing > 0
    warning('%d missing F values across %d subjects x %d models.', ...
        nMissing, numel(subjAll), nModels);
end

fprintf('Extracted F values: %d subjects x %d models.\n', ...
    numel(subjAll), nModels);

end
