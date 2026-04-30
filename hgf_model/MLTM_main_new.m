function MLTM_main_new()
% Master script: invert all candidate models for every subject, then run
% second-level analysis (BMS, parameter extraction, group comparison).
% Requires cleaned data at <repo_root>/cleaned/ (see data_cleaning/).

% Load options
options = MLTM_options_new();

% Step 1: Invert models for all subjects
fprintf('\n Step 1: First-level model inversion \n');
if options.firstlevel(2)
    MLTM_invert_subject_new(options);
end

% Step 2: Second-level analysis (BMS, parameters, group comparison)
fprintf('\n Step 2: Second-level analysis \n');
if any(options.secondlevel)
    MLTM_second_level_new(options);
end

fprintf('\n Analysis complete \n');

end
