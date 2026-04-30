function [groupLabels, groupIdx, metadata_table] = MLTM_load_groups(options)
% Map each subject in options.subjects to its group label (ASD / NT) using
% participant_metadata.csv.
%   groupIdx: 1 = ASD, 2 = NT, NaN = unassigned

if ~exist(options.metadata, 'file')
    error('Metadata file not found: %s', options.metadata);
end

metadata_table = readtable(options.metadata, 'TextType', 'string');

subjectsAll = options.subjects;
nSubjects   = numel(subjectsAll);
groupLabels = cell(nSubjects, 1);
groupIdx    = NaN(nSubjects, 1);

for i = 1:nSubjects
    row = find(metadata_table.participant_id == string(subjectsAll{i}));
    if isempty(row)
        warning('Subject %s not found in metadata.', subjectsAll{i});
        groupLabels{i} = 'UNKNOWN';
        continue;
    end
    grp = char(metadata_table.group(row(1)));
    groupLabels{i} = grp;
    switch grp
        case 'ASD'
            groupIdx(i) = 1;
        case 'NT'
            groupIdx(i) = 2;
        otherwise
            warning('Unexpected group label "%s" for subject %s.', grp, subjectsAll{i});
    end
end

nASD = sum(groupIdx == 1);
nNT  = sum(groupIdx == 2);
fprintf('Group assignment: %d ASD, %d NT (%d unassigned).\n', ...
    nASD, nNT, sum(isnan(groupIdx)));

end
