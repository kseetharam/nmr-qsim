clear;

datapath = path_fix();

addpath('spinach_examples/liquids')
addpath('spinach_examples/solids')
addpath('generator_data/liquids')
addpath('generator_data/solids')

%%%%%%% LIQUIDS

% [spin_system, parameters, H, R, K] = build_noesy_sucrose();
% save(strcat(datapath,'/generators_noesy_sucrose.mat'), 'H', 'R', 'K', 'spin_system', 'parameters','-v7.3')

[spin_system, parameters, H, R, K] = build_noesyhsqc_ubiquitin_deut();
save(strcat(datapath,'/generators_noesyhsqc_ubiquitin_deut.mat'), 'H', 'R', 'K', 'spin_system', 'parameters','-v7.3')

