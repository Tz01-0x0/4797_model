function MLTM_check_correlations_new(options)
% Identifiability check: average each subject's posterior parameter
% correlation matrix using Fisher z (atanh -> mean -> tanh) and flag
% off-diagonal entries with |r| > 0.8.

subjectsAll = options.subjects;
nSubjects   = numel(subjectsAll);
prc_config  = options.model.winningPerceptual;
obs_config  = options.model.winningResponse;

corrMatrices = {};
nCollected   = 0;

for iSubject = 1:nSubjects
    fname = fullfile(options.resultroot, ...
        [subjectsAll{iSubject}, '_', prc_config, '_', obs_config, '.mat']);

    if ~exist(fname, 'file')
        warning('Result file not found for %s', subjectsAll{iSubject});
        continue;
    end

    tmp = load(fname, 'est');

    if isfield(tmp.est, 'optim') && isfield(tmp.est.optim, 'Corr')
        nCollected = nCollected + 1;
        corrMatrices{nCollected, 1} = tmp.est.optim.Corr;
    else
        warning('No correlation matrix found for %s', subjectsAll{iSubject});
    end
end

if nCollected == 0
    warning('No correlation matrices found. Skipping.');
    return;
end

%% Average using Fisher z-transform
nParams = size(corrMatrices{1}, 1);
zSum    = zeros(nParams);

for i = 1:nCollected
    C = corrMatrices{i};
    % Fisher z-transform (atanh), handling diagonal infinities
    Z = real(atanh(C));
    Z(isinf(Z)) = 0;  % diagonal will be inf -> set to 0
    zSum = zSum + Z;
end

zMean   = zSum / nCollected;
avgCorr = tanh(zMean);  % inverse Fisher z

% Restore diagonal
for i = 1:nParams
    avgCorr(i,i) = 1;
end

%% Report
maxCorr = max(avgCorr(~eye(nParams)));
minCorr = min(avgCorr(~eye(nParams)));
fprintf('\n=== Parameter Correlation Check (%d subjects) ===\n', nCollected);
fprintf('  Max off-diagonal correlation: %.4f\n', maxCorr);
fprintf('  Min off-diagonal correlation: %.4f\n', minCorr);

if abs(maxCorr) > 0.8
    warning('High parameter correlation detected (%.3f). Parameters may be unidentifiable.', maxCorr);
end

%% Figure
figure('Name', 'Average Parameter Correlations');
imagesc(avgCorr);
caxis([-1 1]);
colorbar;
colormap(redblue_colormap());
title(sprintf('Average Parameter Correlations (n=%d)', nCollected));

% Label axes with parameter names
paramNames = [options.model.hgfParam, options.model.respParam];
if length(paramNames) == nParams
    set(gca, 'XTick', 1:nParams, 'XTickLabel', paramNames, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:nParams, 'YTickLabel', paramNames);
end

saveas(gcf, fullfile(options.resultroot, 'parameter_correlations.png'));

end


function cmap = redblue_colormap()
% Simple red-white-blue colormap for correlation matrices
n = 128;
r = [linspace(0, 1, n), ones(1, n)];
g = [linspace(0, 1, n), linspace(1, 0, n)];
b = [ones(1, n), linspace(1, 0, n)];
cmap = [r' g' b'];
end
