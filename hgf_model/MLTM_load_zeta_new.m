function [zeta] = MLTM_load_zeta_new(options)
% Collect MAP response-model parameters per subject. Returns nSubjects x 2
% matrix [log(zeta), beta]. zeta is reported in log space; beta is native.

subjectsAll = options.subjects;
nSubjects   = numel(subjectsAll);
mltm_zeta   = NaN(nSubjects, 2);

prc_config = options.model.winningPerceptual;
obs_config = options.model.winningResponse;

for iSubject = 1:nSubjects
    fname = fullfile(options.resultroot, ...
        [subjectsAll{iSubject}, '_', prc_config, '_', obs_config, '.mat']);

    if ~exist(fname, 'file')
        warning('Result file not found for %s', subjectsAll{iSubject});
        continue;
    end

    tmp = load(fname, 'est');

    mltm_zeta(iSubject, 1) = log(tmp.est.p_obs.ze1);
    mltm_zeta(iSubject, 2) = tmp.est.p_obs.beta;
end

zeta = mltm_zeta;

end
