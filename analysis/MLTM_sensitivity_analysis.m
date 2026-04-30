function MLTM_sensitivity_analysis(options)
% Sensitivity check: re-run group comparisons after excluding subjects with
% |z| > 2.5 on any MAP parameter. Outputs full-vs-reduced contrasts on:
%   1. 2x2 ANOVA (Group x Condition)
%   2. Simple t-tests (ASD vs NT)
%   3. Partial correlations AQ ~ parameter | Condition
%   4. GLM AQ ~ parameters + Condition
%
% Usage:
%   MLTM_sensitivity_analysis             % auto-builds options
%   MLTM_sensitivity_analysis(my_opts)    % uses caller-supplied options

Z_THRESHOLD = 2.5;

% Always add hgf_model/ to the path so MLTM_options_new and the loaders
% resolve regardless of which folder the user invokes this script from.
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'hgf_model')));

if nargin < 1
    options = MLTM_options_new();
end

%% Load data
[mltm_par]  = MLTM_load_parameters_new(options);
[mltm_zeta] = MLTM_load_zeta_new(options);
[groupLabels, groupIdx, metadata_table] = MLTM_load_groups(options);

allParams  = [mltm_par, mltm_zeta];
paramNames = [options.model.hgfParam, options.model.respParam];
nParams    = length(paramNames);
nSubj      = size(allParams, 1);

%% Load counterbalancing condition
condIdx = NaN(nSubj, 1);
for i = 1:nSubj
    row = find(metadata_table.participant_id == string(options.subjects{i}));
    if ~isempty(row) && ismember('randomiser_ne38', metadata_table.Properties.VariableNames)
        cond_str = char(metadata_table.randomiser_ne38(row(1)));
        if contains(cond_str, 'Facial', 'IgnoreCase', true)
            condIdx(i) = 1;
        elseif contains(cond_str, 'Verbal', 'IgnoreCase', true)
            condIdx(i) = 2;
        end
    end
end

%% Load AQ scores
aq_scores = NaN(nSubj, 1);
for i = 1:nSubj
    row = find(metadata_table.participant_id == string(options.subjects{i}));
    if ~isempty(row) && ismember('aq_total', metadata_table.Properties.VariableNames)
        aq_scores(i) = metadata_table.aq_total(row(1));
    end
end

%% Identify outliers (|z| > threshold on any parameter)
outlierMask = false(nSubj, 1);
outlierReasons = cell(nSubj, 1);

for p = 1:nParams
    vals = allParams(:, p);
    mu = nanmean(vals);
    sd = nanstd(vals);
    z = abs((vals - mu) / sd);

    for i = 1:nSubj
        if z(i) > Z_THRESHOLD
            outlierMask(i) = true;
            if isempty(outlierReasons{i})
                outlierReasons{i} = sprintf('%s(z=%.1f)', paramNames{p}, z(i));
            else
                outlierReasons{i} = [outlierReasons{i}, ', ', ...
                    sprintf('%s(z=%.1f)', paramNames{p}, z(i))];
            end
        end
    end
end

%% Open log file
logFile = fullfile(options.resultroot, 'sensitivity_analysis_report.txt');
fid = fopen(logFile, 'w');

printBoth(fid, repmat('=', 1, 80));
printBoth(fid, 'SENSITIVITY ANALYSIS');
printBoth(fid, sprintf('Z-score threshold: |z| > %.1f', Z_THRESHOLD));
printBoth(fid, repmat('=', 1, 80));
printBoth(fid, '');

%% Report outliers
printBoth(fid, sprintf('Full sample: N = %d (ASD=%d, NT=%d)', ...
    nSubj, sum(groupIdx==1), sum(groupIdx==2)));
printBoth(fid, sprintf('Outliers identified: N = %d', sum(outlierMask)));

for i = 1:nSubj
    if outlierMask(i)
        gLabel = 'Unknown';
        if groupIdx(i) == 1, gLabel = 'ASD';
        elseif groupIdx(i) == 2, gLabel = 'NT'; end
        printBoth(fid, sprintf('  %s (%s): %s', ...
            options.subjects{i}, gLabel, outlierReasons{i}));
    end
end

reducedMask = ~outlierMask;
printBoth(fid, sprintf('\nReduced sample: N = %d (ASD=%d, NT=%d)', ...
    sum(reducedMask), ...
    sum(groupIdx(reducedMask)==1), ...
    sum(groupIdx(reducedMask)==2)));
printBoth(fid, sprintf('  FF=%d, VF=%d', ...
    sum(condIdx(reducedMask)==1), sum(condIdx(reducedMask)==2)));

%% Run analyses on both samples
sampleNames = {'FULL SAMPLE', 'REDUCED SAMPLE (outliers excluded)'};
sampleMasks = {true(nSubj, 1), reducedMask};

for s = 1:2
    mask = sampleMasks{s};

    printBoth(fid, '');
    printBoth(fid, repmat('=', 1, 80));
    printBoth(fid, sprintf('  %s (N=%d)', sampleNames{s}, sum(mask)));
    printBoth(fid, repmat('=', 1, 80));

    %% 2x2 ANOVA
    printBoth(fid, '');
    printBoth(fid, '--- 2x2 ANOVA: Group x Condition ---');
    printBoth(fid, sprintf('%-12s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s', ...
        'Parameter', 'F_Group', 'p_Group', 'eta2p_G', 'F_Cond', 'p_Cond', 'eta2p_C', 'F_Inter', 'p_Inter', 'eta2p_I'));
    printBoth(fid, repmat('-', 1, 108));

    for p = 1:nParams
        valid = mask & ~isnan(condIdx) & ~isnan(allParams(:,p));
        y = allParams(valid, p);
        g = groupIdx(valid);
        c = condIdx(valid);

        [pvals, tbl] = anovan(y, {g, c}, ...
            'model', 'interaction', ...
            'varnames', {'Group', 'Condition'}, ...
            'display', 'off');

        F_group = tbl{2,6}; p_group = pvals(1);
        F_cond  = tbl{3,6}; p_cond  = pvals(2);
        F_inter = tbl{4,6}; p_inter = pvals(3);

        % Compute partial eta-squared: eta2p = SS_effect / (SS_effect + SS_error)
        SS_group = tbl{2, 2};  SS_cond = tbl{3, 2};  SS_inter = tbl{4, 2};
        SS_error = tbl{5, 2};
        eta2p_group = SS_group / (SS_group + SS_error);
        eta2p_cond  = SS_cond  / (SS_cond  + SS_error);
        eta2p_inter = SS_inter / (SS_inter + SS_error);

        notes = '';
        if p_group < 0.05, notes = [notes ' *G']; end
        if p_cond  < 0.05, notes = [notes ' *C']; end
        if p_inter < 0.05, notes = [notes ' *I']; end

        printBoth(fid, sprintf('%-12s  %8.3f  %8.4f  %8.3f  %8.3f  %8.4f  %8.3f  %8.3f  %8.4f  %8.3f%s', ...
            paramNames{p}, F_group, p_group, eta2p_group, F_cond, p_cond, eta2p_cond, F_inter, p_inter, eta2p_inter, notes));

        % Store for comparison
        res{s}.(paramNames{p}).p_group = p_group;
        res{s}.(paramNames{p}).p_cond  = p_cond;
        res{s}.(paramNames{p}).p_inter = p_inter;
        res{s}.(paramNames{p}).eta2p_group = eta2p_group;
        res{s}.(paramNames{p}).eta2p_cond  = eta2p_cond;
        res{s}.(paramNames{p}).eta2p_inter = eta2p_inter;

        % Simple effects if interaction significant
        if p_inter < 0.05
            printBoth(fid, sprintf('    *** Significant interaction for %s ***', paramNames{p}));
            condLabels = {'Facial-first', 'Verbal-first'};
            for cc = 1:2
                y_asd = allParams(mask & groupIdx==1 & condIdx==cc, p);
                y_nt  = allParams(mask & groupIdx==2 & condIdx==cc, p);
                y_asd = y_asd(~isnan(y_asd));
                y_nt  = y_nt(~isnan(y_nt));
                if length(y_asd) >= 2 && length(y_nt) >= 2
                    [~, pse, ~, sse] = ttest2(y_asd, y_nt);
                    n1 = length(y_asd); n2 = length(y_nt);
                    sp = sqrt(((n1-1)*var(y_asd) + (n2-1)*var(y_nt)) / (n1+n2-2));
                    d = (mean(y_asd) - mean(y_nt)) / sp;
                    printBoth(fid, sprintf('    %s: ASD M=%.4f, NT M=%.4f, t=%.3f, p=%.4f, d=%.3f', ...
                        condLabels{cc}, mean(y_asd), mean(y_nt), sse.tstat, pse, d));
                end
            end
        end
    end

    %% Simple t-tests
    printBoth(fid, '');
    printBoth(fid, '--- Simple t-tests: ASD vs NT ---');
    printBoth(fid, sprintf('%-12s  %8s  %8s  %8s  %8s  %8s  %8s', ...
        'Parameter', 'ASD_M', 'NT_M', 't', 'p', 'd', 'CI95'));
    printBoth(fid, repmat('-', 1, 70));

    for p = 1:nParams
        v_asd = allParams(mask & groupIdx==1, p);
        v_nt  = allParams(mask & groupIdx==2, p);
        v_asd = v_asd(~isnan(v_asd));
        v_nt  = v_nt(~isnan(v_nt));

        [~, pval, ci, st] = ttest2(v_asd, v_nt);
        n1 = length(v_asd); n2 = length(v_nt);
        sp = sqrt(((n1-1)*var(v_asd) + (n2-1)*var(v_nt)) / (n1+n2-2));
        d = (mean(v_asd) - mean(v_nt)) / sp;

        printBoth(fid, sprintf('%-12s  %8.4f  %8.4f  %8.3f  %8.4f  %8.3f  [%.3f,%.3f]', ...
            paramNames{p}, mean(v_asd), mean(v_nt), st.tstat, pval, d, ci(1), ci(2)));

        res{s}.(paramNames{p}).ttest_p = pval;
        res{s}.(paramNames{p}).ttest_d = d;
    end

    %% Partial correlations
    printBoth(fid, '');
    printBoth(fid, '--- Partial Correlation: AQ ~ Parameter | Condition ---');
    printBoth(fid, sprintf('%-12s  %10s  %10s', 'Parameter', 'r_partial', 'p_partial'));
    printBoth(fid, repmat('-', 1, 40));

    cond_dummy = condIdx - nanmean(condIdx(mask));

    for p = 1:nParams
        valid = mask & ~isnan(aq_scores) & ~isnan(allParams(:,p)) & ~isnan(condIdx);
        if sum(valid) > 4
            [r, pval] = partialcorr(aq_scores(valid), allParams(valid,p), cond_dummy(valid));
            sig = '';
            if pval < 0.05, sig = ' *'; end
            printBoth(fid, sprintf('%-12s  %10.4f  %10.4f%s', paramNames{p}, r, pval, sig));

            res{s}.(paramNames{p}).r_partial = r;
            res{s}.(paramNames{p}).p_partial = pval;
        end
    end

    %% GLM (using fitlm for robust R^2 / F / p computation)
    printBoth(fid, '');
    valid_glm = mask & ~isnan(aq_scores) & ~isnan(condIdx);
    for p = 1:nParams
        valid_glm = valid_glm & ~isnan(allParams(:,p));
    end

    if sum(valid_glm) > nParams + 2
        X_tbl = array2table([allParams(valid_glm, :), condIdx(valid_glm)], ...
            'VariableNames', [paramNames, {'Condition'}]);
        X_tbl.AQ = aq_scores(valid_glm);
        mdl = fitlm(X_tbl, 'AQ ~ kappaF + thetaF + kappaV + thetaV + zeta + beta + Condition');
        R2_adj = mdl.Rsquared.Adjusted;
        F_stat = mdl.anova.F(end-1);  % may not be directly available, use Fstat
        % Use overall F-test from the model
        sse = mdl.SSE;
        ssr = mdl.SSR;
        dfr = mdl.NumCoefficients - 1;
        dfe = mdl.DFE;
        F_overall = (ssr/dfr) / (sse/dfe);
        p_overall = 1 - fcdf(F_overall, dfr, dfe);
        R2 = mdl.Rsquared.Ordinary;

        printBoth(fid, sprintf('GLM: R^2=%.3f, R^2_adj=%.3f, F(%d,%d)=%.3f, p=%.4f', ...
            R2, R2_adj, dfr, dfe, F_overall, p_overall));

        res{s}.glm_R2 = R2;
        res{s}.glm_p  = p_overall;
    end
end

%% Robustness comparison
printBoth(fid, '');
printBoth(fid, repmat('=', 1, 80));
printBoth(fid, 'ROBUSTNESS COMPARISON');
printBoth(fid, repmat('=', 1, 80));

% Key findings to compare
keyFindings = {
    'kappaV interaction (Group x Condition)', 'kappaV', 'p_inter';
    'kappaF ~ AQ partial correlation',        'kappaF', 'p_partial';
    'zeta ~ AQ partial correlation',           'zeta',   'p_partial';
    'kappaF Group effect (t-test)',            'kappaF', 'ttest_p';
    'GLM: AQ ~ Parameters',                   '',        '';
};

for k = 1:size(keyFindings, 1)
    printBoth(fid, '');
    printBoth(fid, sprintf('%d. %s:', k, keyFindings{k,1}));

    if k == 5
        % GLM special case
        p_full = res{1}.glm_p;
        p_red  = res{2}.glm_p;
        printBoth(fid, sprintf('   Full:    R^2=%.3f, p=%.4f', res{1}.glm_R2, p_full));
        printBoth(fid, sprintf('   Reduced: R^2=%.3f, p=%.4f', res{2}.glm_R2, p_red));
    else
        pname = keyFindings{k,2};
        fname = keyFindings{k,3};
        p_full = res{1}.(pname).(fname);
        p_red  = res{2}.(pname).(fname);
        printBoth(fid, sprintf('   Full:    p = %.4f', p_full));
        printBoth(fid, sprintf('   Reduced: p = %.4f', p_red));
    end

    if (p_full < 0.05) == (p_red < 0.05)
        printBoth(fid, '   -> ROBUST');
    else
        printBoth(fid, '   -> CHANGED (significance differs)');
    end
end

%% Save
save(fullfile(options.resultroot, 'sensitivity_analysis_results.mat'), 'res', 'outlierMask', 'outlierReasons');
fclose(fid);

printBoth(-1, '');
printBoth(-1, sprintf('Report saved to: %s', logFile));
printBoth(-1, sprintf('Results saved to: %s', fullfile(options.resultroot, 'sensitivity_analysis_results.mat')));

end


function printBoth(fid, str)
% Print to both console and file
fprintf('%s\n', str);
if fid > 0
    fprintf(fid, '%s\n', str);
end
end
