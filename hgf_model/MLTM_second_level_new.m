function MLTM_second_level_new(options)
% Group-level pipeline: BMS -> identifiability check -> parameter extraction
% -> group comparison. Step toggles are options.secondlevel = [a b c d].

fprintf('\n===\n\t Second-Level Analysis Pipeline\n');
fprintf('\t Steps selected:\n');

stepNames = {'Model Comparison (BMS)', ...
             'Check Parameter Correlations', ...
             'Parameter Extraction', ...
             'Group Comparison (ASD vs NT)'};

Analysis_Strategy = options.secondlevel;
for s = 1:length(Analysis_Strategy)
    if s <= length(stepNames)
        status = 'OFF';
        if Analysis_Strategy(s), status = 'ON'; end
        fprintf('\t  [%s] %s\n', status, stepNames{s});
    end
end
fprintf('===\n\n');
pause(1);

doModelComparison            = Analysis_Strategy(1);
doCheckParameterCorrelations = Analysis_Strategy(2);
doParameterExtraction        = Analysis_Strategy(3);
doGroupComparison            = Analysis_Strategy(4);

%% Step 1: Model Comparison (BMS)
if doModelComparison
    fprintf('\n--- Step 1: Model Comparison ---\n');
    MLTM_model_selection_new(options);
end

%% Step 2: Check Parameter Correlations
if doCheckParameterCorrelations
    fprintf('\n--- Step 2: Parameter Correlations ---\n');
    MLTM_check_correlations_new(options);
end

%% Step 3: Parameter Extraction
if doParameterExtraction
    fprintf('\n--- Step 3: Parameter Extraction ---\n');
    MLTM_extract_parameters_new(options);
end

%% Step 4: Group Comparison (ASD vs NT)
if doGroupComparison
    fprintf('\n--- Step 4: Group Comparison ---\n');
    MLTM_group_comparison(options);
end

fprintf('\n=== Second-Level Analysis Complete ===\n');

end
