function c = softmax_facial_verbal_config

% Configuration for the softmax response model for facial-verbal cue integration.
%
% Free parameters:
%   zeta (ze1): facial cue weight bias. In log-space.
%               ze1 > 1 => facial precision upweighted
%               ze1 < 1 => verbal precision upweighted
%   beta:       inverse temperature / softmax sharpness. In log-space.
%
% Adapted from softmax_reward_social_config.m
% --------------------------------------------------------------------------------
% Original Copyright (C) 2012 Christoph Mathys, Andreea Diaconescu TNU, UZH & ETHZ

c = struct;
c.model = 'softmax_facial_verbal';

% Zeta in log-space
% Prior: log(ze1) ~ N(log(e), 5^2) => ze1 prior mean ~ e ~ 2.72
c.logze1mu = log(2.7183);
c.logze1sa = 5^2;

% Beta in log-space
% Prior: log(beta) ~ N(log(48), 5^2) => beta prior mean ~ 48
c.logbetamu = log(48);
c.logbetasa = 5^2;

% Gather prior settings in vectors
c.priormus = [
    c.logze1mu,...
    c.logbetamu,...
    ];

c.priorsas = [
    c.logze1sa,...
    c.logbetasa,...
    ];

% Model function handle
c.obs_fun = @softmax_facial_verbal;

% Handle to function that transforms observation parameters
c.transp_obs_fun = @softmax_facial_verbal_transp;

return;
