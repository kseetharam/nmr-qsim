% 1H-1H NOESY spectrum of GB1 with everything deuterated except 
% methyl groups. Deuteria are kept in the spin system because
% they are a part of the coupling network; methyl group rotati-
% on is not accounted for in this simulation.
%
% Calculation time: hours.
%
% i.kuprov@soton.ac.uk

%function methyl_noesy_gb1_mod()
diary('output.txt');
% Protein data import
options.pdb_mol=1;
options.select='all';
options.noshift='delete';
options.deuterate='non-Me';
options.deuterate={'non-Me','LEU','VAL','ILE'}; %these are the minimal number of residues to track the fold-unfolding 
                                                %transitions according to
                                                %experiments

%options.deuterate={'non-Me','LEU'};


% options.deuterate={'non-Me','ALA','MET'};
%[sys,inter,aux]=protein_mod_lam('2N9K.pdb','2N9K.bmrb',options);
[sys,inter]=protein_mod('5JXV.pdb','5JXV.bmrb',options);


% Magnet field
sys.magnet=21.1356;

% Tolerances
sys.tols.inter_cutoff=100;              % Only significant DD couplings
sys.tols.prox_cutoff=5.0;               % Increase till convergence

% Relaxation theory
%{
inter.relaxation={'redfield'};
inter.rlx_keep='kite';
inter.equilibrium='zero';
inter.tau_c={5e-9};
%}

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='IK-1';
bas.connectivity='scalar_couplings';
bas.level=2; bas.space_level=2;         % Pairwise approximation

% Algorithmic options
sys.enable={'prop_cache','op_cache','greedy'};

% Create the spin system structure
spin_system=create(sys,inter);


% Kill carbons and nitrogens (protein assumed unlabelled)
%spin_system=kill_spin(spin_system,strcmp('13C',spin_system.comp.isotopes));
spin_system=kill_spin(spin_system,strcmp('15N',spin_system.comp.isotopes));

% % Kill deuterons
spin_system=kill_spin(spin_system,strcmp('2H',spin_system.comp.isotopes));

% Build the basis
spin_system=basis(spin_system,bas);



%subsystems=dilute(spin_system,'13C');
save gb1_LEU_VAL_ILE.mat spin_system

% Sequence parameters
parameters.J=140; %by taking a look to the output of the extraction-parameters subroutine and checking couplings for residues
%parameters.tmix=200e-3;
%parameters.offset=[750 0];
parameters.offset=[5000 0];
%parameters.offset=[0 0];
parameters.sweep=[12500 4000];
%parameters.npoints=[1024 1024];
%parameters.npoints=[1024 1024];
%parameters.zerofill=[2048 2048];

parameters.npoints = [512 512];
parameters.zerofill = [1024 1024];

%parameters.spins={'1H','13C'};
parameters.spins={'13C','1H'};
parameters.axis_units='ppm';
%parameters.axis_units='Hz';
parameters.decouple_f1={'1H'};
parameters.decouple_f2={'13C'};
parameters.rho0=state(spin_system,'Lz','1H','cheap');

%{
%%testing the dilution feature for 13C....
spectrum=zeros(parameters.zerofill(2),...
               parameters.zerofill(1),'like',1i);

% Loop over isotopomers
parfor n=1:numel(subsystems)
    
    % Build the basis
    subsystem=basis(subsystems{n},bas);
   
    % Simulation
    fid=liquid(subsystem,@hmqc,parameters,'nmr');
    
    % Apodisation
    fid=apodisation(spin_system,fid,{{'cos'},{'cos'}});
    
    % Fourier transform
    spectrum=spectrum+fftshift(fft2(fid,parameters.zerofill(2),...
                                        parameters.zerofill(1)));
    
end
%}


% Simulation
%fid=liquid(spin_system,@noesy,parameters,'nmr');
fid=liquid(spin_system,@hmqc,parameters,'nmr');

% Apodisation
%fid.cos=apodisation(spin_system,fid.cos,{{'sqcos'},{'sqcos'}});
%fid.sin=apodisation(spin_system,fid.sin,{{'sqcos'},{'sqcos'}});
%fid=apodisation(spin_system,fid,{{'sqcos'},{'sqcos'}});

% F2 Fourier transform
%f1_cos=real(fftshift(fft(fid.cos,parameters.zerofill(2),1),1));
%f1_sin=real(fftshift(fft(fid.sin,parameters.zerofill(2),1),1));

% States signal
%f1_states=f1_cos-1i*f1_sin;

% F1 Fourier transform
%spectrum=fftshift(fft(f1_states,parameters.zerofill(1),2),2);
%spectrum = fftshift(fft(fid,parameters.zerofill(2),1),1);
%spectrum = fftshift(fft(spectrum,parameters.zerofill(2),2),2);

spectrum=fftshift(fft2(fid,parameters.zerofill(2),parameters.zerofill(1)));

% Plotting
figure(); scale_figure([1.5 2.0]);
plot_2d(spin_system,abs(spectrum),parameters,...
        20,[0.00125 0.0125 0.00125 0.0125],2,256,6,'positive');

diary off
%}
%end

