% NOESY spectrum of sucrose (magnetic parameters computed with DFT).
%
% Calculation time: minutes
%
% i.kuprov@soton.ac.uk
% luke.edwards@ucl.ac.uk

function [spin_system, parameters, H, R, K] = noesy_sucrose()

% Spin system properties (vacuum DFT calculation)
options.min_j=1.0;
[sys,inter]=g2spinach(gparse('sucrose.log'),...
                                     {{'H','1H'}},31.8,options);
% Magnet field
sys.magnet=5.9;

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='IK-2';
bas.connectivity='scalar_couplings';
bas.space_level=3;

% Relaxation theory parameters
inter.relaxation={'redfield'};
inter.equilibrium='IME';
inter.temperature=298;
inter.rlx_keep='kite';
inter.tau_c={200e-12};

% Algorithmic options
sys.enable={'greedy'};
sys.disable={'krylov'};
sys.tols.prox_cutoff=4.0;

% Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

% Sequence parameters
parameters.tmix=0.5;
parameters.offset=800;
parameters.sweep=[1700 1700];
parameters.npoints=[512 512];
parameters.zerofill=[2048 2048];
parameters.spins={'1H'};
parameters.axis_units='ppm';
parameters.needs={'rho_eq'};

% Simulation
[H, R, K]=generator_liquid(spin_system,@noesy,parameters,'nmr');

end

