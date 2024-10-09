% NOESY spectrum of strychnine.
%
% Calculation time: minutes
%
% i.kuprov@soton.ac.uk
% luke.edwards@ucl.ac.uk

function [spin_system, parameters, H, R, K, rhot, rho0, obs_p, obs_z] = dyn_noesy_strychnine()

% Spin system properties
[sys,inter]=strychnine({'1H'});

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
parameters.npoints_mix = 100;
parameters.offset=1200;
parameters.sweep=[2500 2500];
parameters.npoints=[512 512];
parameters.zerofill=[2048 2048];
parameters.spins={'1H'};
parameters.axis_units='ppm';
parameters.needs={'rho_eq'};

% Simulation
[rhot, rho0, obs_p, obs_z, H, R, K]=dyn_liquid(spin_system,@noesy_trajectory,parameters,'nmr');

end

