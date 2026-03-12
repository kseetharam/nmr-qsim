% ZULF spectrum of Gemcitabine with 13C at natural abundace 
% all 19F & 1H
% Florin, 03/02/2026
clear all; 

carbon_list=[14,5,17]; %Index of C atoms closest to 19F atoms in the pentose ring
k=1;
% Load Gaussian16 Output
[sys,inter1]=g2spinach(gparse('gemcitabine.log'),{{'H','1H'},{'F','19F'},{'C','13C'},{'O','17O'},{'N','15N'}},[31.8 100 300 500 400]);
sys.magnet=500*1e-9;
sys.output='hush';
sys.disable={'hygiene'};

% index=[15,16,22,23,28,carbon_list(k)]; %short list: 2x19F, 3x1H, 1x13C
idxF=find(sys.isotopes == "19F");
idxH=[20 21 22 23 24 25 28]; %all non-exchangeable 1H spins
index=[idxF,idxH,carbon_list(k)]; %long list all 19F & 1H
sys.isotopes=[repelem({'19F','1H'}, [numel(idxF), numel(idxH)]),{'13C'}]; % 10 spins 

inter.coupling.scalar=[];
for i=1:size(sys.isotopes,2)
    for j=1:size(sys.isotopes,2)
        inter.coupling.scalar{i,j}=1/2*inter1.coupling.scalar{index(i),index(j)};
    end
end
% inter.coordinates=inter1.coordinates(index);

% Basis set
bas.formalism='sphten-liouv';
%bas.approximation='none';
bas.approximation = 'IK-0';
bas.level = 4;
%bas.connectivity='scalar_couplings';

% Relaxation theory parameters
inter.relaxation={'redfield'};
inter.equilibrium='zero';%'dibari';
inter.rlx_keep='labframe';
inter.tau_c={100e-12};
inter.temperature=298;

% Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

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

%Saving spin system data for post-processing...
filename = sprintf('./gemcitabine_spin_system_IK0_4.mat');

save(filename,'spin_system');



% Plotting
figure(1); hold on; 
x0 = spin('1H')*sys.magnet/(2*pi);
xline(x0,'--','LineWidth',1.5);
plot_1d(spin_system,real(spectrum(k,:))',parameters,'LineWidth',2); hold on;
xlabel('ZULF spectrum (Hz)');
set(gca,'FontSize',40)
