% 1H-1H NOESY spectrum of ubiquitin with 65 ms mixing time. It is
% assumed that the protein is not 13C- or 15N-labelled.
%
% Calculation time: hours.
%
% luke.edwards@ucl.ac.uk
% i.kuprov@soton.ac.uk

function [spin_system, parameters, H, R, K] = build_noesy_ubiquitin()

% Protein data import
options.pdb_mol=1;
options.select='all';
options.noshift='delete';
[sys,inter]=protein('1D3Z.pdb','1D3Z.bmrb',options);

% Magnet field
sys.magnet=21.1356;

% Tolerances
sys.tols.inter_cutoff=2.0;
sys.tols.prox_cutoff=4.0;
sys.enable={'greedy'};

% Relaxation theory
inter.relaxation={'redfield'};
inter.rlx_keep='kite';
inter.equilibrium='zero';
inter.tau_c={5e-9};

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='IK-1';
bas.connectivity='scalar_couplings';
bas.level=4; bas.space_level=3;

% Create the spin system structure
spin_system=create(sys,inter);

% Kill carbons and nitrogens (protein assumed unlabelled)
spin_system=kill_spin(spin_system,strcmp('13C',spin_system.comp.isotopes));
spin_system=kill_spin(spin_system,strcmp('15N',spin_system.comp.isotopes));

% Build the basis
spin_system=basis(spin_system,bas);

% Sequence parameters
parameters.tmix=0.065;
parameters.offset=4250;
parameters.sweep=[11750 11750];
parameters.npoints=[512 512];
parameters.zerofill=[2048 2048];
parameters.spins={'1H'};
parameters.axis_units='ppm';
parameters.rho0=state(spin_system,'Lz','1H','cheap');

report(spin_system,'building the generators...');

% Build generators
[H, R, K] =generator_liquid(spin_system,@noesy,parameters,'nmr');

end

