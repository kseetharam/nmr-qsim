% ZULF spectrum of Gemcitabine: exact 10-spin Hilbert-space reference
% Formalism: zeeman-hilb  (Hilbert dim = 2^10 = 1024 vs Liouville 4^10 = 1M)
% Memory: ~few GB vs ~100 GB for sphten-liouv/none
% 10 spins: 2x19F + 7x1H + 1x13C  (same subset as gemcitabine_ZULF_13C_single.m)
clear all;

carbon_list=[14,5,17];
k=1;

[sys,inter1]=g2spinach(gparse('gemcitabine.log'),{{'H','1H'},{'F','19F'},{'C','13C'},{'O','17O'},{'N','15N'}},[31.8 100 300 500 400]);
sys.magnet=500*1e-9;
sys.output='hush';
sys.disable={'hygiene'};

idxF=find(sys.isotopes == "19F");
idxH=[20 21 22 23 24 25 28];
index=[idxF,idxH,carbon_list(k)];
sys.isotopes=[repelem({'19F','1H'}, [numel(idxF), numel(idxH)]),{'13C'}];

inter.coupling.scalar=[];
for i=1:size(sys.isotopes,2)
    for j=1:size(sys.isotopes,2)
        inter.coupling.scalar{i,j}=1/2*inter1.coupling.scalar{index(i),index(j)};
    end
end
for i=1:numel(index)
    inter.coordinates{i}   = inter1.coordinates{index(i)};
    inter.zeeman.matrix{i} = inter1.zeeman.matrix{index(i)};
end

% Hilbert-space basis: exact, no truncation
bas.formalism='zeeman-hilb';
bas.approximation='none';

% Relaxation parameters (identical to Liouville version for fair comparison)
inter.relaxation={'redfield'};
inter.equilibrium='zero';
inter.rlx_keep='labframe';
inter.tau_c={100e-12};
inter.temperature=298;

spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

weights=spin_system.inter.gammas/spin('1H');

% Initial state: gamma-weighted Lz sum (density matrix in Hilbert space)
rho_sud=sparse(0);
for n=1:spin_system.comp.nspins
    rho_sud=rho_sud+weights(n)*state(spin_system,{'Lz'},{n});
end

% Detection operators: operator() in Hilbert space because FID = Tr[coil * rho(t)]
% (contrast with Liouville space where coil is a bra vector built with state())
coilZ=sparse(0); coilXY=sparse(0); coilXYquad=sparse(0);
for n=1:spin_system.comp.nspins
    coilZ=coilZ+weights(n)*operator(spin_system,{'Lz'},{n});
    coilXY=coilXY+weights(n)*(operator(spin_system,{'L+'},{n})+operator(spin_system,{'L-'},{n}))/2;
    coilXYquad=coilXYquad+weights(n)*operator(spin_system,{'L+'},{n});
end

% Pulse operators (operator() is correct in both formalisms)
Sx=sparse(0); Sy=sparse(0); Sz=sparse(0);
for n=1:spin_system.comp.nspins
    Sx=Sx+weights(n)*(operator(spin_system,{'L+'},{n})+operator(spin_system,{'L-'},{n}))/2;
    Sy=Sy+weights(n)*(operator(spin_system,{'L+'},{n})-operator(spin_system,{'L-'},{n}))/2i;
    Sz=Sz+weights(n)*operator(spin_system,{'Lz'},{n});
end

% Single-pulse experiment
parameters.offset=0;
parameters.sweep=3000;
parameters.npoints=4*1024;
parameters.zerofill=4*1024;
parameters.rho0=step(spin_system,Sy,rho_sud,pi/2);
parameters.coil=coilXYquad;
parameters.spins={'1H'};
parameters.invert_axis=0;
parameters.axis_units='Hz';

fid=liquid(spin_system,@acquire,parameters,'labframe');
fid_raw=fid;

fid=apodisation(spin_system,fid,{{'exp',3}});
spectrum=fftshift(fft(fid,parameters.zerofill));
freq=(-parameters.zerofill/2 : parameters.zerofill/2-1)*(parameters.sweep/parameters.zerofill);

% Save to distinct path (does not overwrite Liouville-space results)
zulf_data_path='../../scripts/zulf_numerics/data/gemcitabine_ZULF_13C_hilb_exact_fid_spectrum.mat';
spec_real=real(spectrum); spec_imag=imag(spectrum);
fid_raw_real=real(fid_raw); fid_raw_imag=imag(fid_raw);
save(zulf_data_path,'fid_raw_real','fid_raw_imag','spec_real','spec_imag','freq','-v7.3');
fprintf('Saved exact Hilbert-space FID -> %s\n', zulf_data_path);

figure(1); hold on;
x0=spin('1H')*sys.magnet/(2*pi);
xline(x0,'--','LineWidth',1.5);
plot_1d(spin_system,real(spectrum)',parameters,'LineWidth',2);
xlabel('ZULF spectrum (Hz)');
set(gca,'FontSize',40);
