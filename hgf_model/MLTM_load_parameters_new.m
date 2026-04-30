function [perceptualParameters] = MLTM_load_parameters_new(options)
% Collect MAP perceptual parameters [kappaF, thetaF, kappaV, thetaV] from the
% winning model fits for every subject. Returns nSubjects x 4 matrix.

subjectsAll = options.subjects;
nSubjects   = numel(subjectsAll);
mltm_par    = NaN(nSubjects, 4);

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

    mltm_par(iSubject, 1) = tmp.est.p_prc.ka_f;
    mltm_par(iSubject, 2) = tmp.est.p_prc.th_f;
    mltm_par(iSubject, 3) = tmp.est.p_prc.ka_v;
    mltm_par(iSubject, 4) = tmp.est.p_prc.th_v;
end

perceptualParameters = mltm_par;

end
