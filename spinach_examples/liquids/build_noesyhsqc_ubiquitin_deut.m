% 1H-1H-15N NOESY-HSQC spectrum of 15N-labelled ubiquitin at 900
% MHz with 65 ms mixing time. It is assumed that the protein is 
% not 13C-labelled. Specific positions are deuterated, and deu-
% terium nuclei are simulated explicitly.
%
% Calculation time: a week on 32 cores, needs 512 GB of RAM.
%
% luke.edwards@ucl.ac.uk
% i.kuprov@soton.ac.uk
% andras_boeszoermenyi@hms.harvard.edu

function [spin_system, parameters, H, R, K] = noesyhsqc_ubiquitin_deut()

% Protein data import
options.pdb_mol=1;
options.select='all';
options.noshift='delete';
options.deuterate={'HA','HB','HB1','HB2','HB3','HG','HG1','HG2','HG3',...
                   'HD','HD1','HD2','HD3','HE','HE1','HE2','HE3','HZ',...
                   'HZ1','HZ2','HZ3','HH','HH1','HH2','HH3'};
[sys,inter]=protein('1D3Z.pdb','1D3Z.bmrb',options);

% Magnet field
sys.magnet=21.1356;

% Tolerances
sys.tols.inter_cutoff=2.0;
sys.tols.prox_cutoff=4.0;
sys.enable={'greedy'};

% Relaxation theory
inter.relaxation={'redfield','t1_t2'};
inter.r1_rates=num2cell(100*strcmp('2H',sys.isotopes));
inter.r2_rates=num2cell(100*strcmp('2H',sys.isotopes));
inter.rlx_keep='kite';
inter.equilibrium='zero';
inter.tau_c={1e-8};

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='IK-1';
bas.connectivity='scalar_couplings';
bas.level=4; bas.space_level=3;

% Create the spin system structure
spin_system=create(sys,inter);

% Kill carbons
spin_system=kill_spin(spin_system,strcmp('13C',spin_system.comp.isotopes));

% Build the basis
spin_system=basis(spin_system,bas);

% Sequence parameters
parameters.tmix=0.090;
parameters.J=90.0;
parameters.npoints=[128 64 128];
parameters.zerofill=[512 256 512];
parameters.spins={'1H','15N','1H'};
parameters.offset=[4250 -10600 4250];
parameters.sweep=[10750 3000 10750];
parameters.axis_units='ppm';

% Simulation
[H, R, K]=generator_liquid(spin_system,@noesyhsqc,parameters,'nmr');

end

