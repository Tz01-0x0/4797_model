function MLTM_group_comparison(options)
% Parameter-level group analysis (called from MLTM_second_level_new):
%   1. 2x2 ANOVA Group(ASD/NT) x Condition(Facial-first/Verbal-first), Type III SS
%   2. ASD vs NT t-tests + Cohen's d on each parameter
%   3. partialcorr(AQ, parameter | condition)
%   4. Boxplots per parameter and per Group x Condition cell
% Produces Table 4 of the report.

%% Load data
[mltm_par]  = MLTM_load_parameters_new(options);
[mltm_zeta] = MLTM_load_zeta_new(options);
[groupLabels, groupIdx, metadata_table] = MLTM_load_groups(options);

allParams  = [mltm_par, mltm_zeta];
paramNames = [options.model.hgfParam, options.model.respParam];
nParams    = length(paramNames);

idxASD = groupIdx == 1;
idxNT  = groupIdx == 2;

if sum(idxASD) < 2 || sum(idxNT) < 2
    error('Not enough subjects in one or both groups (ASD=%d, NT=%d).', ...
        sum(idxASD), sum(idxNT));
end

%% Load counterbalancing condition
condIdx = NaN(length(options.subjects), 1);  % 1=Facial First, 2=Verbal First
for i = 1:length(options.subjects)
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

nFF = sum(condIdx == 1);
nVF = sum(condIdx == 2);
fprintf('\nCounterbalancing: Facial-first=%d, Verbal-first=%d, unassigned=%d\n', ...
    nFF, nVF, sum(isnan(condIdx)));

% Cross-tabulation
fprintf('  ASD: FF=%d, VF=%d\n', sum(idxASD & condIdx==1), sum(idxASD & condIdx==2));
fprintf('  NT:  FF=%d, VF=%d\n', sum(idxNT  & condIdx==1), sum(idxNT  & condIdx==2));

%% 2x2 ANOVA: Group x Condition (primary analysis)
fprintf('\n=== 2x2 ANOVA: Group (ASD/NT) x Condition (FF/VF) ===\n');
fprintf('%-12s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n', ...
    'Parameter', 'F_Group', 'p_Group', 'eta2p_G', 'F_Cond', 'p_Cond', 'eta2p_C', 'F_Inter', 'p_Inter', 'eta2p_I');
fprintf('%s\n', repmat('-', 1, 108));

results = struct();
validSubj = ~isnan(condIdx);  % exclude subjects without condition info

for p = 1:nParams
    y    = allParams(validSubj, p);
    g    = groupIdx(validSubj);     % 1=ASD, 2=NT
    cond = condIdx(validSubj);      % 1=FF, 2=VF

    % Remove NaN parameter values
    keep = ~isnan(y);
    y = y(keep); g = g(keep); cond = cond(keep);

    % Two-way ANOVA (unbalanced: use anovan with type III SS)
    [pvals, tbl, stats_anova] = anovan(y, {g, cond}, ...
        'model', 'interaction', ...
        'varnames', {'Group', 'Condition'}, ...
        'display', 'off');

    F_group = tbl{2, 6};  p_group = pvals(1);
    F_cond  = tbl{3, 6};  p_cond  = pvals(2);
    F_inter = tbl{4, 6};  p_inter = pvals(3);

    % Compute partial eta-squared: eta2p = SS_effect / (SS_effect + SS_error)
    SS_group = tbl{2, 2};  SS_cond = tbl{3, 2};  SS_inter = tbl{4, 2};
    SS_error = tbl{5, 2};
    eta2p_group = SS_group / (SS_group + SS_error);
    eta2p_cond  = SS_cond  / (SS_cond  + SS_error);
    eta2p_inter = SS_inter / (SS_inter + SS_error);

    fprintf('%-12s  %8.3f  %8.4f  %8.3f  %8.3f  %8.4f  %8.3f  %8.3f  %8.4f  %8.3f', ...
        paramNames{p}, F_group, p_group, eta2p_group, F_cond, p_cond, eta2p_cond, F_inter, p_inter, eta2p_inter);
    if p_group < 0.05, fprintf('  *G'); end
    if p_cond  < 0.05, fprintf('  *C'); end
    if p_inter < 0.05, fprintf('  *I'); end
    fprintf('\n');

    results.(paramNames{p}).F_group = F_group;
    results.(paramNames{p}).p_group = p_group;
    results.(paramNames{p}).eta2p_group = eta2p_group;
    results.(paramNames{p}).F_cond  = F_cond;
    results.(paramNames{p}).p_cond  = p_cond;
    results.(paramNames{p}).eta2p_cond = eta2p_cond;
    results.(paramNames{p}).F_inter = F_inter;
    results.(paramNames{p}).p_inter = p_inter;
    results.(paramNames{p}).eta2p_inter = eta2p_inter;
    results.(paramNames{p}).anova_tbl = tbl;

    % If interaction significant, report simple effects
    if p_inter < 0.05
        fprintf('    *** Significant interaction for %s ***\n', paramNames{p});
        for c = 1:2
            cond_label = {'Facial-first', 'Verbal-first'};
            y_asd_c = allParams(idxASD & condIdx == c, p);
            y_nt_c  = allParams(idxNT  & condIdx == c, p);
            y_asd_c = y_asd_c(~isnan(y_asd_c));
            y_nt_c  = y_nt_c(~isnan(y_nt_c));
            if length(y_asd_c) >= 2 && length(y_nt_c) >= 2
                [~, pval_se, ~, stats_se] = ttest2(y_asd_c, y_nt_c);
                n1 = length(y_asd_c); n2 = length(y_nt_c);
                sp = sqrt(((n1-1)*var(y_asd_c) + (n2-1)*var(y_nt_c)) / (n1+n2-2));
                d_se = (mean(y_asd_c) - mean(y_nt_c)) / sp;
                fprintf('    %s: ASD M=%.4f, NT M=%.4f, t=%.3f, p=%.4f, d=%.3f\n', ...
                    cond_label{c}, mean(y_asd_c), mean(y_nt_c), ...
                    stats_se.tstat, pval_se, d_se);
            end
        end
    end
end

%% Simple t-tests (for reference / backward compatibility)
fprintf('\n=== Simple t-tests: ASD vs NT (ignoring condition) ===\n');
fprintf('%-12s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n', ...
    'Parameter', 'ASD_M', 'ASD_SD', 'NT_M', 'NT_SD', 't-stat', 'p-value', 'd');
fprintf('%s\n', repmat('-', 1, 80));

for p = 1:nParams
    vals_asd = allParams(idxASD, p);
    vals_nt  = allParams(idxNT, p);
    vals_asd = vals_asd(~isnan(vals_asd));
    vals_nt  = vals_nt(~isnan(vals_nt));

    [~, pval, ci, stats_t] = ttest2(vals_asd, vals_nt);

    n1 = length(vals_asd); n2 = length(vals_nt);
    sp = sqrt(((n1-1)*var(vals_asd) + (n2-1)*var(vals_nt)) / (n1+n2-2));
    cohens_d = (mean(vals_asd) - mean(vals_nt)) / sp;

    fprintf('%-12s  %8.4f  %8.4f  %8.4f  %8.4f  %8.3f  %8.4f  %8.3f', ...
        paramNames{p}, mean(vals_asd), std(vals_asd), ...
        mean(vals_nt), std(vals_nt), stats_t.tstat, pval, cohens_d);
    if pval < 0.05, fprintf('  *'); end
    if pval < 0.01, fprintf('*'); end
    fprintf('\n');

    results.(paramNames{p}).ttest_asd_mean = mean(vals_asd);
    results.(paramNames{p}).ttest_nt_mean  = mean(vals_nt);
    results.(paramNames{p}).ttest_tstat    = stats_t.tstat;
    results.(paramNames{p}).ttest_pval     = pval;
    results.(paramNames{p}).ttest_cohens_d = cohens_d;
    results.(paramNames{p}).ttest_ci       = ci;
end

%% AQ analyses (with condition controlled)
aq_scores = NaN(length(options.subjects), 1);
for i = 1:length(options.subjects)
    row = find(metadata_table.participant_id == string(options.subjects{i}));
    if ~isempty(row) && ismember('aq_total', metadata_table.Properties.VariableNames)
        aq_scores(i) = metadata_table.aq_total(row(1));
    end
end

if any(~isnan(aq_scores))
    aq_asd = aq_scores(idxASD); aq_asd = aq_asd(~isnan(aq_asd));
    aq_nt  = aq_scores(idxNT);  aq_nt  = aq_nt(~isnan(aq_nt));
    if ~isempty(aq_asd) && ~isempty(aq_nt)
        [~, pval_aq, ~, stats_aq] = ttest2(aq_asd, aq_nt);
        fprintf('\nAQ Total: ASD M=%.2f (SD=%.2f), NT M=%.2f (SD=%.2f), t=%.3f, p=%.6f\n', ...
            mean(aq_asd), std(aq_asd), mean(aq_nt), std(aq_nt), stats_aq.tstat, pval_aq);
    end

    %% Partial correlation: AQ ~ Parameter, controlling for Condition
    fprintf('\n=== Partial Correlation: AQ ~ Parameter | Condition ===\n');
    fprintf('%-12s  %8s  %8s  %8s  %8s\n', 'Parameter', 'r_partial', 'p_partial', 'r_simple', 'p_simple');

    % Condition dummy for partial correlation (mean-centred)
    cond_dummy = condIdx - nanmean(condIdx);

    for p = 1:nParams
        valid = ~isnan(aq_scores) & ~isnan(allParams(:,p)) & ~isnan(condIdx);
        if sum(valid) > 4
            % Simple correlation
            [r_simple, p_simple] = corr(aq_scores(valid), allParams(valid, p));

            % Partial correlation controlling for condition
            [r_partial, p_partial] = partialcorr(aq_scores(valid), allParams(valid, p), ...
                cond_dummy(valid));

            fprintf('%-12s  %8.4f  %8.4f  %8.4f  %8.4f', ...
                paramNames{p}, r_partial, p_partial, r_simple, p_simple);
            if p_partial < 0.05, fprintf('  *'); end
            fprintf('\n');

            results.(paramNames{p}).r_partial = r_partial;
            results.(paramNames{p}).p_partial = p_partial;
            results.(paramNames{p}).r_simple  = r_simple;
            results.(paramNames{p}).p_simple  = p_simple;
        end
    end

    %% GLM: AQ ~ HGF Parameters + Condition (using fitlm, not regress)
    % NOTE: regress() was replaced with fitlm() because SPM's regress()
    % can shadow MATLAB's built-in version and produce invalid statistics.
    fprintf('\n=== GLM: AQ ~ HGF Parameters + Condition ===\n');
    valid_glm = ~isnan(aq_scores) & ~isnan(condIdx);
    for p = 1:nParams
        valid_glm = valid_glm & ~isnan(allParams(:,p));
    end

    if sum(valid_glm) > nParams + 2
        X_tbl = array2table([allParams(valid_glm, :), condIdx(valid_glm)], ...
            'VariableNames', [paramNames, {'Condition'}]);
        X_tbl.AQ = aq_scores(valid_glm);
        mdl = fitlm(X_tbl, 'AQ ~ kappaF + thetaF + kappaV + thetaV + zeta + beta + Condition');

        R2 = mdl.Rsquared.Ordinary;
        R2_adj = mdl.Rsquared.Adjusted;
        sse = mdl.SSE;
        ssr = mdl.SSR;
        dfr = mdl.NumCoefficients - 1;
        dfe = mdl.DFE;
        F_overall = (ssr/dfr) / (sse/dfe);
        p_overall = 1 - fcdf(F_overall, dfr, dfe);

        fprintf('  R^2=%.3f, R^2_adj=%.3f, F(%d,%d)=%.3f, p=%.4f\n', ...
            R2, R2_adj, dfr, dfe, F_overall, p_overall);

        % Print individual coefficients
        coefNames = mdl.CoefficientNames;
        coefEst   = mdl.Coefficients.Estimate;
        coefP     = mdl.Coefficients.pValue;
        fprintf('  Coefficients:\n');
        for c = 1:length(coefNames)
            sig = '';
            if coefP(c) < 0.05, sig = ' *'; end
            if coefP(c) < 0.01, sig = ' **'; end
            fprintf('    %-12s  b=%.4f  p=%.4f%s\n', coefNames{c}, coefEst(c), coefP(c), sig);
        end

        results.glm_Rsq = R2;
        results.glm_Rsq_adj = R2_adj;
        results.glm_F   = F_overall;
        results.glm_p   = p_overall;
        results.glm_model = mdl;
    end
end

%% Figures: Box plots by group x condition
figure('Name', 'Group x Condition Comparison', 'Position', [100 100 1400 500]);
for p = 1:nParams
    subplot(1, nParams, p);

    % Create 4-group labels: ASD-FF, ASD-VF, NT-FF, NT-VF
    grp4 = NaN(size(allParams, 1), 1);
    grp4(idxASD & condIdx == 1) = 1;  % ASD Facial-first
    grp4(idxASD & condIdx == 2) = 2;  % ASD Verbal-first
    grp4(idxNT  & condIdx == 1) = 3;  % NT Facial-first
    grp4(idxNT  & condIdx == 2) = 4;  % NT Verbal-first

    valid = ~isnan(grp4) & ~isnan(allParams(:,p));
    boxplot(allParams(valid, p), grp4(valid), ...
        'Labels', {'ASD-FF', 'ASD-VF', 'NT-FF', 'NT-VF'});

    title(paramNames{p});
    p_g = results.(paramNames{p}).p_group;
    p_i = results.(paramNames{p}).p_inter;
    subtitle(sprintf('G:p=%.3f, I:p=%.3f', p_g, p_i));
end
sgtitle('Group x Condition: Model Parameters');
saveas(gcf, fullfile(options.resultroot, 'group_condition_boxplots.png'));

% Also keep original 2-group boxplot
figure('Name', 'Group Comparison', 'Position', [100 100 1200 400]);
for p = 1:nParams
    subplot(1, nParams, p);
    grp_data = [allParams(idxASD, p); allParams(idxNT, p)];
    grp_labels = [ones(sum(idxASD), 1); 2*ones(sum(idxNT), 1)];
    boxplot(grp_data, grp_labels, 'Labels', {'ASD', 'NT'});
    title(paramNames{p});
    d = results.(paramNames{p}).ttest_cohens_d;
    pval = results.(paramNames{p}).ttest_pval;
    subtitle(sprintf('p=%.3f, d=%.2f', pval, d));
end
sgtitle('ASD vs NT: Model Parameters');
saveas(gcf, fullfile(options.resultroot, 'group_comparison_boxplots.png'));

%% Save results
save(fullfile(options.resultroot, 'group_comparison_results.mat'), 'results', '-mat');
fprintf('\nGroup comparison results saved.\n');

end
