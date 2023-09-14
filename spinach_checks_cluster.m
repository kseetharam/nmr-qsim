clear;

datapath = path_fix();
% addpath('spinach_examples/liquids/')
% addpath('spinach_examples/solids/')
addpath('/n/holyscratch01/jaffe_lab/Everyone/kis/nmr-qsim/spinach_examples/liquids/')
addpath('/n/holyscratch01/jaffe_lab/Everyone/kis/nmr-qsim/spinach_examples/solids/')


% datapath = '../data/generator_data/liquids/';
% % addpath('spinach_examples/liquids/')
% % addpath('spinach_examples/solids/')
% addpath('/Users/kis/KIS Dropbox/Kushal Seetharam/NMR QSim/Code/nmr-qsim/spinach_examples/liquids/')
% addpath('/Users/kis/KIS Dropbox/Kushal Seetharam/NMR QSim/Code/nmr-qsim/spinach_examples/solids/')

%%%%%%% LIQUIDS

% [spin_system, parameters, H, R, K] = build_noesy_sucrose();
% save(strcat(datapath,'generators_noesy_sucrose.mat'), 'H', 'R', 'K', 'spin_system', 'parameters','-v7.3')

[spin_system, parameters, H, R, K] = build_noesyhsqc_ubiquitin_deut();
save(strcat(datapath,'generators_noesyhsqc_ubiquitin_deut.mat'), 'H', 'R', 'K', 'spin_system', 'parameters','-v7.3')

