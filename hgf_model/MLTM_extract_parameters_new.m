function MLTM_extract_parameters_new(options)
% Pool MAP parameters from the winning model into a per-subject table
% (results/MLTM_MAP_estimates.csv) and run a simple AQ ~ parameters GLM.

%% Load parameters
[mltm_par]  = MLTM_load_parameters_new(options);
[mltm_zeta] = MLTM_load_zeta_new(options);
[groupLabels, groupIdx, metadata_table] = MLTM_load_groups(options);

subjectsAll = options.subjects;
nSubjects   = numel(subjectsAll);

%% Get AQ scores from metadata
aq_scores = NaN(nSubjects, 1);
for i = 1:nSubjects
    row = find(metadata_table.participant_id == string(subjectsAll{i}));
    if ~isempty(row) && ismember('aq_total', metadata_table.Properties.VariableNames)
        aq_scores(i) = metadata_table.aq_total(row(1));
    end
end

%% Print parameter summary
paramNames = [options.model.hgfParam, options.model.respParam];
allParams  = [mltm_par, mltm_zeta];
fprintf('\n=== Parameter Summary (Winning Model) ===\n');
fprintf('%-12s  %8s  %8s  %8s  %8s\n', 'Parameter', 'Mean', 'SD', 'Min', 'Max');
for p = 1:length(paramNames)
    vals = allParams(:, p);
    fprintf('%-12s  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
        paramNames{p}, nanmean(vals), nanstd(vals), nanmin(vals), nanmax(vals));
end

%% GLM: AQ ~ perceptual params + zeta + intercept
if any(~isnan(aq_scores))
    design_matrix = [mltm_par, mltm_zeta(:,1), ones(nSubjects, 1)];
    validRows = ~any(isnan([aq_scores, design_matrix]), 2);

    if sum(validRows) > size(design_matrix, 2)
        [B, ~, ~, ~, stats] = regress(aq_scores(validRows), design_matrix(validRows,:));
        fprintf('\nGLM: AQ ~ kappaF + thetaF + kappaV + thetaV + log(zeta) + intercept\n');
        fprintf('  R^2 = %.4f, F = %.4f, p = %.6f\n', stats(1), stats(2), stats(3));
        fprintf('  Coefficients: ');
        coeffLabels = [options.model.hgfParam, {'log_zeta', 'intercept'}];
        for c = 1:length(B)
            fprintf('%s=%.4f  ', coeffLabels{c}, B(c));
        end
        fprintf('\n');
    else
        fprintf('\nNot enough valid rows for GLM (AQ ~ parameters).\n');
    end
end

%% Save as table
columnNames = [{'participant_id', 'group'}, options.model.hgfParam, ...
               options.model.respParam, {'aq_total'}];

tableData = cell(nSubjects, length(columnNames));
for i = 1:nSubjects
    tableData{i, 1} = subjectsAll{i};
    tableData{i, 2} = groupLabels{i};
    for p = 1:size(allParams, 2)
        tableData{i, 2 + p} = allParams(i, p);
    end
    tableData{i, end} = aq_scores(i);
end

t = cell2table(tableData, 'VariableNames', columnNames);

ofile = fullfile(options.resultroot, 'MLTM_MAP_estimates.csv');
writetable(t, ofile);
fprintf('\nParameter table saved to: %s\n', ofile);

end
