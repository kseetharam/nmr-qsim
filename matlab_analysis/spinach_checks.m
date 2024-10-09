clear;

datapath = '/Users/kis/KIS Dropbox/Kushal Seetharam/NMR QSim/Code/data/';
addpath(genpath('/Users/kis/KIS Dropbox/Kushal Seetharam/NMR QSim/Code/nmr-qsim/spinach_examples/liquids/'))
addpath(genpath('/Users/kis/KIS Dropbox/Kushal Seetharam/NMR QSim/Code/nmr-qsim/spinach_examples/solids/'))

%%%%%%% LIQUIDS

% [spin_system, parameters, H, R, K] = build_hcch_tocsy_gb1();
% save(strcat(datapath,'generator_data/liquids/generators_hcch_tocsy_gb1.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesy_methanol();
% save(strcat(datapath,'generator_data/liquids/generators_noesy_methanol.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesy_strychnine();
% save(strcat(datapath,'generator_data/liquids/generators_noesy_strychnine.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesy_sucrose();
% save(strcat(datapath,'generator_data/liquids/generators_noesy_sucrose1.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesy_ubiquitin();
% save(strcat(datapath,'generator_data/liquids/generators_noesy_ubiquitin.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

[spin_system, parameters, H, R, K] = build_noesyhsqc_ubiquitin_deut();
save(strcat(datapath,'generator_data/liquids/generators_noesyhsqc_ubiquitin_deut.mat'), 'H', 'R', 'K', 'spin_system', 'parameters','-v7.3')

%%%%%%% SOLIDS

% [spin_system, parameters, H, R, K] = build_static_powder_trp();
% save(strcat(datapath,'generator_data/solids/generators_static_powder_trp.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_static_powder_suc();
% save(strcat(datapath,'generator_data/solids/generators_static_powder_suc.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_static_powder_gly();
% save(strcat(datapath,'generator_data/solids/generators_static_powder_gly.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_mas_powder_suc_floquet();
% save(strcat(datapath,'generator_data/solids/generators_mas_powder_suc_floquet.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_mas_powder_suc_gridfree();
% save(strcat(datapath,'generator_data/solids/generators_mas_powder_suc_gridfree.mat'), 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H1, R1, K1, H2, R2, K2] = build_mas_powder_trp_floquet();
% save(strcat(datapath,'generator_data/solids/generators_mas_powder_trp_floquet.mat'), 'H1', 'R1', 'K1', 'H2', 'R2', 'K2', 'spin_system', 'parameters')

% [spin_system, parameters, H1, R1, K1, H2, R2, K2] = build_mas_powder_trp_gridfree();
% save(strcat(datapath,'generator_data/solids/generators_mas_powder_trp_gridfree.mat'), 'H1', 'R1', 'K1', 'H2', 'R2', 'K2', 'spin_system', 'parameters', '-v7.3')
