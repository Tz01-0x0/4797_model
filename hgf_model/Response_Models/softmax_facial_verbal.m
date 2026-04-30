function logp = softmax_facial_verbal(r, infStates, ptrans)
% Response model: precision-weighted integration of facial and verbal cue
% beliefs into a left/right choice. Card values are NOT used (choice is made
% before card reveal). Beta is symmetrically modulated by both Level-3
% volatility estimates (both cues are social):
%   v_diff   = 2*b_left - 1
%   beta_eff = beta * exp(-muhat3_f - muhat3_v)
%   P(left)  = sigmoid(beta_eff * v_diff)
% Free params: zeta (log-space facial-vs-verbal weight bias), beta.

% Transform parameters to native space
ze1  = exp(ptrans(1));   % zeta: facial weight bias
beta = exp(ptrans(2));   % inverse temperature

% Number of trials
n = size(infStates,1);

% Initialize log-probabilities as NaN (NaN for irregular trials)
logp = NaN(n,1);

% Extract beliefs from perceptual mode
% x_f = P(facial cue is valid) = muhat_1 for facial stream
x_f = infStates(:,1,1);
% x_v = P(verbal cue is valid) = muhat_1 for verbal stream
x_v = infStates(:,1,3);

% Level-3 predictions (volatility estimates) for beta modulation
mu3hat_f = infStates(:,3,1);   % facial volatility belief
mu3hat_v = infStates(:,3,3);   % verbal volatility belief

% Extract cue directions from input matrix
facial_left  = r.u(:,3);   % 1 if facial cue points left
verbal_left  = r.u(:,4);   % 1 if verbal cue points left
% Card values (cols 5-6) are NOT used: participants choose before seeing
% card values, so they cannot influence the decision.

% Remove irregular trials
x_f(r.irr) = [];
x_v(r.irr) = [];
mu3hat_f(r.irr) = [];
mu3hat_v(r.irr) = [];
facial_left(r.irr)  = [];
verbal_left(r.irr)  = [];

y = r.y(:,1);
y(r.irr) = [];

% Compute precision weights
% Precision at Level 1 (Fisher information of Bernoulli)
pf = 1./(x_f.*(1-x_f));    % facial precision
pv = 1./(x_v.*(1-x_v));    % verbal precision

% Precision-weighted combination with zeta bias
% ze1 > 1: facial cue gets more weight
% ze1 < 1: verbal cue gets more weight
wf = ze1.*pf ./ (ze1.*pf + pv);    % facial weight
wv = pv ./ (ze1.*pf + pv);          % verbal weight

% Translate beliefs into choice space
% P(correct side is left | facial cue):
%   If facial points left AND cue is believed valid: high P(left)
%   If facial points left AND cue is believed invalid: low P(left)
%   = facial_left * x_f + (1 - facial_left) * (1 - x_f)
p_left_facial = facial_left .* x_f + (1 - facial_left) .* (1 - x_f);

% P(correct side is left | verbal cue):
p_left_verbal = verbal_left .* x_v + (1 - verbal_left) .* (1 - x_v);

% Precision-weighted belief about left being correct
b_left = wf .* p_left_facial + wv .* p_left_verbal;

% Volatility-modulated inverse temperature
% - Adapted from Sevgi et al. (Version 3) with SYMMETRIC modulation.
%   Both cues are social, so both contribute equally:
%     beta_eff = beta * exp(-mu3hat_f) * exp(-mu3hat_v)
%   Higher perceived volatility in either stream => lower effective
%   beta => more exploratory choices.
beta = exp(-mu3hat_f - mu3hat_v + log(beta));

% Pure cue-belief decision variable
% - No card values: participants choose before seeing card points.
% - v_diff > 0 when belief favours left, < 0 when it favours right.
v_diff = 2 .* b_left - 1;

% Probability of choosing left
prob = 1 ./ (1 + exp(-beta .* v_diff .* (2.*y - 1)));

% Clamp to avoid log(0)
prob = max(prob, 1e-10);
prob = min(prob, 1 - 1e-10);

% Assign to non-irregular trials
reg = ~ismember(1:n, r.irr);
logp(reg) = log(prob);

return;
