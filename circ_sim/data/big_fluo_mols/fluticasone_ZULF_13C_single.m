% ZULF spectrum of Fluticasone with 13C at natural abundace 
% all 19F & 1H
% Florin, 03/02/2026
clear all; 

carbon_list=[7, 17, 36]; %Index of C atoms closest to 19F atoms in the pentose ring
k=1;
% Load Gaussian16 Output
[sys,inter1]=g2spinach(gparse('fluticasone.log'),{{'H','1H'},{'F','19F'},{'C','13C'},{'O','17O'},{'N','15N'},{'S','32S'}},[31.8 100 300 500 400 400]);
sys.magnet=500*1e-9;
sys.output='hush';
sys.disable={'hygiene'};

%index=[8, 18, 37, 48, 65, 66, carbon_list(k)];% short list 3x19F,3x1H,1x13C
idxF=find(sys.isotopes == "19F");
idxH=find(sys.isotopes == "1H");% all 1H spins 
idxH(idxH==56)=[]; % all non-exchangeable 1H spins
index=[idxF,idxH,carbon_list(k)]; %long list all 19F & 1H
sys.isotopes=[repelem({'19F','1H'}, [numel(idxF), numel(idxH)]),{'13C'}]; % 32 spins 

inter.coupling.scalar=[];
for i=1:size(sys.isotopes,2)
    for j=1:size(sys.isotopes,2)
        inter.coupling.scalar{i,j}=1/2*inter1.coupling.scalar{index(i),index(j)};
    end
end
inter.coordinates=inter1.coordinates(index);

% =========================================================================
% Save molecular parameters for Python Lindblad simulation
% Output: circ_sim/scripts/linblad_dyn/data/fluticasone_params.mat
% =========================================================================
N_sel = numel(index);

% J-coupling upper triangular matrix (Hz).
% inter.coupling.scalar already has the 1/2 factor applied (same convention
% as the gemcitabine parameters validated against Spinach).
J_hz_upper = zeros(N_sel, N_sel);
for ii = 1:N_sel
    for jj = (ii+1):N_sel
        val = inter.coupling.scalar{ii,jj};
        if ~isempty(val)
            J_hz_upper(ii,jj) = val;
        end
    end
end

% Nuclear coordinates (Angstrom) — N x 3
coords_ang = zeros(N_sel, 3);
for ii = 1:N_sel
    coords_ang(ii,:) = inter.coordinates{ii};
end

% Full Zeeman chemical shielding tensors (ppm) — N x 3 x 3.
% inter1.zeeman.matrix{k} is the 3x3 DFT shielding tensor for full-system
% spin k; we select the subset via index().
sigma_ppm_3d = zeros(N_sel, 3, 3);
for ii = 1:N_sel
    zm = inter1.zeeman.matrix{index(ii)};
    if ~isempty(zm)
        sigma_ppm_3d(ii,:,:) = zm;
    end
end

% Isotope labels as comma-separated string (e.g. '19F,19F,1H,...,13C')
isotope_str = strjoin(sys.isotopes, ',');

% Gyromagnetic ratios (rad/s/T) in spin order — extracted from Spinach
gammas_rad = zeros(1, N_sel);
for ii = 1:N_sel
    gammas_rad(ii) = spin(sys.isotopes{ii});
end

out_path = fullfile('..', '..', 'scripts', 'linblad_dyn', 'data', 'fluticasone_params.mat');
out_dir   = fileparts(out_path);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

save(out_path, 'J_hz_upper', 'coords_ang', 'sigma_ppm_3d', 'isotope_str', 'gammas_rad', '-v7');
fprintf('Saved molecular parameters -> %s\n', out_path);
fprintf('  N_spins : %d\n', N_sel);
fprintf('  Spin order: %s\n', isotope_str);
zm0 = squeeze(sigma_ppm_3d(1,:,:));
fprintf('  sigma_ppm_3d check (spin 0): iso=%.2f ppm, max_aniso=%.2f ppm\n', ...
        trace(zm0)/3, max(max(abs(zm0 - trace(zm0)/3*eye(3)))));
% =========================================================================

% Basis set
bas.formalism='sphten-liouv';
%bas.approximation='none';
bas.approximation = 'IK-0';
bas.level = 2;

% Relaxation theory parameters
inter.relaxation={'redfield'};
inter.equilibrium='zero';%'dibari';
inter.rlx_keep='labframe';
inter.tau_c={100e-12};
inter.temperature=298;

% Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

filename = sprintf('./fluticasone_spin_system_IK0_4.mat');

save(filename,'spin_system');
disp('Spin system stored')
% Magnetogyric ratio weights relative to 1H
weights=spin_system.inter.gammas/spin('1H');

% Get gamma-weighted initial state (sudden transfer)
rho_sud=sparse(0);
for n=1:spin_system.comp.nspins
    rho_sud=rho_sud+weights(n)*state(spin_system,{'Lz'},{n});
end

% Get gamma-weighted detection state
coilZ=sparse(0);coilXY=sparse(0);coilXYquad=sparse(0);
for n=1:spin_system.comp.nspins
    coilZ=coilZ+weights(n)*state(spin_system,{'Lz'},{n});
    coilXY=coilXY+weights(n)*(state(spin_system,{'L+'},{n})+state(spin_system,{'L-'},{n}))/2;
    coilXYquad=coilXYquad+weights(n)*state(spin_system,{'L+'},{n});
end

% Get gamma-weighted pulse operator
Sx=sparse(0); Sy=sparse(0); Sz=sparse(0);
for n=1:spin_system.comp.nspins
    Sx=Sx+weights(n)*(operator(spin_system,{'L+'},{n})+...
                      operator(spin_system,{'L-'},{n}))/2;
    Sy=Sy+weights(n)*(operator(spin_system,{'L+'},{n})-...
                      operator(spin_system,{'L-'},{n}))/2i;
    Sz=Sz+weights(n)*operator(spin_system,{'Lz'},{n});
end

R=relaxation(spin_system);

% Simulation of the single pulse experiment
parameters.offset = 0;            % observation offset (Hz)
parameters.sweep = 3000; % sweep width (Hz)
parameters.npoints = 4*1024;      % number of FID points
parameters.zerofill = 4*1024;
parameters.rho0 = step(spin_system,Sy,rho_sud,pi/2); %apply 90deg pulse on Y axis
parameters.coil = coilXYquad; %quadrature detection
parameters.spins={'1H'};
parameters.invert_axis=0;
parameters.axis_units='Hz';

fid=liquid(spin_system,@acquire,parameters,'labframe');

% Apodization
% fid=apodization(fid,'exp-1d',3);
fid=apodisation(spin_system,fid,{{'exp',3}});

% Fourier transform
spectrum(k,:)=fftshift(fft(fid,parameters.zerofill));

% Plotting
figure(1); hold on; 
x0 = spin('1H')*sys.magnet/(2*pi); 
xline(x0,'--','LineWidth',1.5);
plot_1d(spin_system,real(spectrum(k,:))',parameters,'LineWidth',2); hold on;
xlabel('ZULF spectrum (Hz)');
set(gca,'FontSize',40)
