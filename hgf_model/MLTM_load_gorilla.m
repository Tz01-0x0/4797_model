function [responses, inputs, metadata] = MLTM_load_gorilla(csv_path)
% Read a cleaned trial-level CSV and return the (responses, inputs, metadata)
% triple expected by fitModel.
%
% inputs columns:
%   1: facial_cue_valid       4: verbal_direction_left
%   2: verbal_cue_valid       5: card_left_points
%   3: facial_direction_left  6: card_right_points
%   7: equivalence_group (1=A, 2=B, 3=C)

% Read CSV
T = readtable(csv_path);

% Response vector
% response_for_model: chose_left (1/0), NaN for excluded trials
responses = T.response_for_model;

% Input matrix
% Encode cue directions as numeric
facial_dir_left = double(strcmp(T.facial_cue_direction, 'left'));
verbal_dir_left = double(strcmp(T.verbal_cue_direction, 'left'));

inputs = [T.facial_cue_valid, ...      % col 1
          T.verbal_cue_valid, ...       % col 2
          facial_dir_left, ...          % col 3
          verbal_dir_left, ...          % col 4
          T.card_left_points, ...       % col 5
          T.card_right_points, ...      % col 6
          T.eg_numeric];               % col 7: equivalence group

% Metadata
metadata = struct;
metadata.trial_order = T.trial_order;
metadata.trial_number = T.trial_number;
metadata.phase = T.phase;
metadata.phase_numeric = T.phase_numeric;
metadata.equivalence_group = T.equivalence_group;
metadata.eg_numeric = T.eg_numeric;
metadata.rt = T.rt_raw;
metadata.is_timeout = T.is_timeout;
metadata.chose_left = T.chose_left;
metadata.followed_facial = T.followed_facial;
metadata.followed_verbal = T.followed_verbal;

fprintf('Loaded %d trials from %s\n', length(responses), csv_path);
fprintf('  Valid responses: %d\n', sum(~isnan(responses)));
fprintf('  NaN (excluded):  %d\n', sum(isnan(responses)));

return;
