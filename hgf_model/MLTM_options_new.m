function options = MLTM_options_new()
% Central configuration: paths, model space, parameter names, analysis flags.
% Subject list is auto-detected from data/cleaned/*_hgf_input.csv.

%% Paths
options.code       = fullfile(fileparts(mfilename('fullpath')));
options.dataroot   = fullfile(options.code, '..', 'data', 'cleaned');
options.resultroot = fullfile(options.code, 'results');
if ~exist(options.resultroot, 'dir'); mkdir(options.resultroot); end

addpath(genpath(fullfile(options.code, 'Perceptual_Models')));
addpath(genpath(fullfile(options.code, 'Response_Models')));
addpath(genpath(fullfile(options.code, 'Inversion')));
addpath(genpath(fullfile(options.code, 'Optimization')));
addpath(genpath(fullfile(options.code, 'Utils')));

%% Model space
% Arch A: unified facial HGF (3 EGs collapsed, prereg baseline)
% Arch B: 3 independent facial HGFs sharing kappa_f / theta_f
% Arch C: shared x3_f, separate x2_f per EG
% M4    : Arch A + free generalisation parameter gamma (exploratory)
options.model.perceptualModels = {
    'hgf_binary3l_facial_verbal_config', ...        % Arch A
    'hgf_binary3l_facial_verbal_gamma_config', ...  % M4
    'hgf_binary3l_facial_verbal_archB_config', ...  % Arch B
    'hgf_binary3l_facial_verbal_archC_config', ...  % Arch C
};
options.model.responseModels = { 'softmax_facial_verbal_config' };
options.model.labels = {'M1_ArchA', 'M4_GammaFree', 'ArchB', 'ArchC'};

% Set after BMS (Arch A wins in our data)
options.model.winningPerceptual = 'hgf_binary3l_facial_verbal_config';
options.model.winningResponse   = 'softmax_facial_verbal_config';

%% Parameter names extracted from the winning model
options.model.hgfParam  = {'kappaF','thetaF','kappaV','thetaV'};
options.model.respParam = {'zeta','beta'};

%% Bundled SPM subset (spm_BMS + dependencies; see hgf_model/spm/README.md)
options.spmpath = fullfile(options.code, 'spm');
if exist(options.spmpath, 'dir')
    addpath(options.spmpath);
    if exist('spm_BMS', 'file') ~= 2
        warning('spm_BMS not found after addpath(%s). Check the spm folder.', options.spmpath);
    end
else
    warning('Bundled spm folder not found at %s.', options.spmpath);
end

%% Group metadata (produced by data_cleaning/comprehensive_cleaning_pipeline.py)
options.metadata = fullfile(options.code, '..', 'data', 'participant_metadata.csv');

%% Analysis flags
options.firstlevel  = [1 1];      % [loadData, invert]
options.secondlevel = [1 1 1 1];  % [BMS, correlationCheck, paramExtract, groupComparison]

%% Auto-detect subject IDs from cleaned/*_hgf_input.csv
% The cleaning pipeline writes a *_hgf_input.csv for every participant whose
% raw task data is present, INCLUDING ones that should be excluded from the
% analysis (e.g. 100% timeout). The canonical inclusion list is participant_
% metadata.csv — we intersect the two so re-cleaning from raw cannot silently
% reintroduce excluded participants into BMS / group analyses.
csv_files = dir(fullfile(options.dataroot, '*_hgf_input.csv'));
all_in_cleaned = cell(1, length(csv_files));
for i = 1:length(csv_files)
    name = csv_files(i).name;
    all_in_cleaned{i} = name(1:end-length('_hgf_input.csv'));
end

if exist(options.metadata, 'file') == 2
    meta_tbl = readtable(options.metadata, 'TextType', 'string');
    valid_ids = string(meta_tbl.participant_id);
    keep = ismember(string(all_in_cleaned), valid_ids);
    options.subjects = all_in_cleaned(keep);
    n_dropped = sum(~keep);
    if n_dropped > 0
        dropped = all_in_cleaned(~keep);
        fprintf(['MLTM_options_new: dropped %d subject(s) present in cleaned/ ' ...
                 'but absent from participant_metadata.csv: %s\n'], ...
                n_dropped, strjoin(dropped, ', '));
    end
else
    warning(['participant_metadata.csv not found at %s — falling back to ' ...
             'every *_hgf_input.csv file. Excluded participants will be ' ...
             'fitted unless removed by hand.'], options.metadata);
    options.subjects = all_in_cleaned;
end

if isempty(options.subjects)
    warning('No cleaned data files in %s. Run the data_cleaning pipeline first.', options.dataroot);
end
fprintf('Found %d subjects in %s\n', length(options.subjects), options.dataroot);

end
