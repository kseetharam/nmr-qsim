clear;

addpath('spinach_examples/liquids')
addpath('spinach_examples/solids')
addpath('generator_data/liquids')
addpath('generator_data/solids')

%%%%%%% LIQUIDS

% [spin_system, parameters, H, R, K] = build_hcch_tocsy_gb1();
% save('generator_data/liquids/generators_hcch_tocsy_gb1.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesy_methanol();
% save('generator_data/liquids/generators_noesy_methanol.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesy_strychnine();
% save('generator_data/liquids/generators_noesy_strychnine.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

[spin_system, parameters, H, R, K] = build_noesy_sucrose();
save('generator_data/liquids/generators_noesy_sucrose.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesy_ubiquitin();
% save('generator_data/liquids/generators_noesy_ubiquitin.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_noesyhsqc_ubiquitin_deut();
% save('generator_data/liquids/generators_noesyhsqc_ubiquitin_deut.mat', 'H', 'R', 'K', 'spin_system', 'parameters','-v7.3')

%%%%%%% SOLIDS

% [spin_system, parameters, H, R, K] = build_static_powder_trp();
% save('generator_data/solids/generators_static_powder_trp.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_static_powder_suc();
% save('generator_data/solids/generators_static_powder_suc.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_static_powder_gly();
% save('generator_data/solids/generators_static_powder_gly.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_mas_powder_suc_floquet();
% save('generator_data/solids/generators_mas_powder_suc_floquet.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H, R, K] = build_mas_powder_suc_gridfree();
% save('generator_data/solids/generators_mas_powder_suc_gridfree.mat', 'H', 'R', 'K', 'spin_system', 'parameters')

% [spin_system, parameters, H1, R1, K1, H2, R2, K2] = build_mas_powder_trp_floquet();
% save('generator_data/solids/generators_mas_powder_trp_floquet.mat', 'H1', 'R1', 'K1', 'H2', 'R2', 'K2', 'spin_system', 'parameters')

% [spin_system, parameters, H1, R1, K1, H2, R2, K2] = build_mas_powder_trp_gridfree();
% save('generator_data/solids/generators_mas_powder_trp_gridfree.mat', 'H1', 'R1', 'K1', 'H2', 'R2', 'K2', 'spin_system', 'parameters', '-v7.3')
