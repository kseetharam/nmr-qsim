clc
clear all

props=gparse('alanine.log');
[sys,inter]=g2spinach(props,{{'H','1H'}},31.8);
H_index = [2 3 4 5];

sys.isotopes = {'1H','1H','1H','1H'};
coordinates = cell(4,1);
for i = 1:4
    coordinates{i} = inter.coordinates{H_index(i)};
end
inter.coordinates = coordinates;

inter = rmfield(inter,'zeeman');
inter.zeeman.scalar={3.69 1.39 1.39 1.39};


coupling = cell(4);
for i = 1:4
    for j = 1:4
        coupling{i,j} = inter.coupling.scalar{H_index(i),H_index(j)};
    end
end
inter.coupling.scalar = coupling;
JM = (coupling{2,3} + coupling{2,4} + coupling{3,4})/3;
JMH = (coupling{1,2} + coupling{1,3} + coupling{1,4})/3;
[inter.coupling.scalar{1,2:4}] = deal(JMH);
inter.coupling.scalar{2,3} = JM;
inter.coupling.scalar{2,4} = JM;
inter.coupling.scalar{3,4} = JM;
for i = 1:4
    for j = 1:4
        inter.coupling.scalar{j,i} = inter.coupling.scalar{i,j};
    end
end

% Magnet field
sys.magnet=18.78;

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='none';
bas.sym_group={'S3'};
bas.sym_spins={[2 3 4]};

% Relaxation theory parameters
inter.relaxation={'redfield'};
inter.equilibrium='dibari';
inter.temperature=300;
inter.rlx_keep='kite';
inter.tau_c={0.05e-9};

% Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

% experimental
parameters.tmix=0.8;
parameters.offset= 3759.37;
parameters.sweep=[6398.95 9615.38];
parameters.npoints=[512 512];
% parameters.zerofill=2*[2048 2048];
parameters.zerofill=2*[512 512];
parameters.spins={'1H'};
parameters.axis_units='ppm';
parameters.rho0=state(spin_system,'Lz','1H','chem');


%% Original
[fid,H,R]=liquid_mod(spin_system,@noesy,parameters,'nmr');

% Apodization
fid.cos=apodization(fid.cos,'sqcosbell-2d');
fid.sin=apodization(fid.sin,'sqcosbell-2d');
% fid.cos=apodisation(spin_system,fid.cos,{{'sqcos'},{'sqcos'}});
% fid.sin=apodisation(spin_system,fid.sin,{{'sqcos'},{'sqcos'}});

% F2 Fourier transform
f1_cos=real(fftshift(fft(fid.cos,parameters.zerofill(2),1),1));
f1_sin=real(fftshift(fft(fid.sin,parameters.zerofill(2),1),1));

% States signal
f1_states=f1_cos-1i*f1_sin;

% F1 Fourier transform
spectrum=fftshift(fft(f1_states,parameters.zerofill(1),2),2);

% Plotting
figure(); scale_figure([1.5 2.0]);
plot_2d(spin_system,-real(spectrum),parameters,...
        20,[0.02 0.2 0.02 0.2],2,256,6,'both');

%% Brute force
dt=1./parameters.sweep;
Lx=operator(spin_system,'Lx',parameters.spins{1});
Ly=operator(spin_system,'Ly',parameters.spins{1});
Lz=operator(spin_system,'Lz',parameters.spins{1});

coil=state(spin_system,'L+',parameters.spins{1},'cheap');

% first 90x pulse
U90x = expm(-1i*Lx*pi/2);
rho_initial = U90x*parameters.rho0;
dim = length(parameters.rho0);
% t1 evolution
L_net=H+1i*R;
L_dt1 = expm(-1i*L_net*dt(1));
rho_stack = zeros(dim,parameters.npoints(1));
rho_stack(:,1) = rho_initial;
rho_temp = rho_initial;

for i=2:parameters.npoints(1)
   rho_temp=L_dt1*rho_temp;
   rho_stack(:,i) = rho_temp;
end

pulse_90x = expm(-1i*Lx*pi/2);
pulse_90y = expm(-1i*Ly*pi/2);
pulse_90mx = expm(1i*Lx*pi/2);
pulse_90my = expm(1i*Ly*pi/2);
pulse_mix = expm(-1i*L_net*parameters.tmix);

% second 90 deg pulse (90x, 90y, -90x, -90y)
rho_stack1 = zeros(dim,parameters.npoints(1),4);
Mat1 = pulse_90x;
Mat2 = pulse_90y;
Mat3 = pulse_90mx;
Mat4 = pulse_90my;
for i = 1:parameters.npoints(1)
    rho_stack1(:,i,1) = Mat1*rho_stack(:,i);
    rho_stack1(:,i,2) = Mat2*rho_stack(:,i);
    rho_stack1(:,i,3) = Mat3*rho_stack(:,i);
    rho_stack1(:,i,4) = Mat4*rho_stack(:,i);
end

% gradient
Npts_grad = 100;
phi = linspace(0,2*pi,Npts_grad);

gradmat = zeros(dim,dim,Npts_grad);
for k = 1:Npts_grad
    gradmat(:,:,k) =  expm(-1i*full(Lz)*phi(k));
end

for j=1:size(rho_stack1,2)
    rho_temp = rho_stack1(:,j,1);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,1) = rho_sum;

    rho_temp = rho_stack1(:,j,2);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,2) = rho_sum;

    rho_temp = rho_stack1(:,j,3);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,3) = rho_sum;

    rho_temp = rho_stack1(:,j,4);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,4) = rho_sum;

end


% mixing
for i = 1:parameters.npoints(1)
    rho_stack1(:,i,1) = pulse_mix*rho_stack1(:,i,1);
    rho_stack1(:,i,2) = pulse_mix*rho_stack1(:,i,2);
    rho_stack1(:,i,3) = pulse_mix*rho_stack1(:,i,3);
    rho_stack1(:,i,4) = pulse_mix*rho_stack1(:,i,4);
end


% gradient
Npts_grad = 100;
phi = linspace(0,2*pi,Npts_grad);
for j=1:size(rho_stack1,2)
    rho_temp = rho_stack1(:,j,1);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,1) = rho_sum;

    rho_temp = rho_stack1(:,j,2);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,2) = rho_sum;

    rho_temp = rho_stack1(:,j,3);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,3) = rho_sum;

    rho_temp = rho_stack1(:,j,4);
    rho_sum = 0;
    for i = 1:Npts_grad
        rho_sum = rho_sum + gradmat(:,:,i)*rho_temp;
    end
    rho_sum = rho_sum/Npts_grad;
    rho_stack1(:,j,4) = rho_sum;

end

% third 90 deg pulse (90y)

for i = 1:parameters.npoints(1)
    rho_stack1(:,i,1) = pulse_90y*rho_stack1(:,i,1);
    rho_stack1(:,i,2) = pulse_90y*rho_stack1(:,i,2);
    rho_stack1(:,i,3) = pulse_90y*rho_stack1(:,i,3);
    rho_stack1(:,i,4) = pulse_90y*rho_stack1(:,i,4);
end


% calculate fid
L_dt2 = expm(-1i*L_net*dt(2));
fid_temp = zeros(parameters.npoints(2),parameters.npoints(1),4);
for i = 1:parameters.npoints(1)
    rho1 = rho_stack1(:,i,1);
    rho2 = rho_stack1(:,i,2);
    rho3 = rho_stack1(:,i,3);
    rho4 = rho_stack1(:,i,4);    
    for j = 1:parameters.npoints(2)
        fid_temp(j,i,1) = trace(coil'*rho1);
        rho1 = L_dt2*rho1;

        fid_temp(j,i,2) = trace(coil'*rho2);
        rho2 = L_dt2*rho2;

        fid_temp(j,i,3) = trace(coil'*rho3);
        rho3 = L_dt2*rho3;

        fid_temp(j,i,4) = trace(coil'*rho4);
        rho4 = L_dt2*rho4;

    end
end

fid_brute.cos = fid_temp(:,:,1) - fid_temp(:,:,3);
fid_brute.sin = fid_temp(:,:,2) - fid_temp(:,:,4);


% Apodization
fid_brute.cos=apodization(fid_brute.cos,'sqcosbell-2d');
fid_brute.sin=apodization(fid_brute.sin,'sqcosbell-2d');
% fid_brute.cos=apodisation(spin_system,fid.cos,{{'sqcos'},{'sqcos'}});
% fid_brute.nsin=apodisation(spin_system,fid.sin,{{'sqcos'},{'sqcos'}});

% F2 Fourier transform
f1_cos=real(fftshift(fft(fid_brute.cos,parameters.zerofill(2),1),1));
f1_sin=real(fftshift(fft(fid_brute.sin,parameters.zerofill(2),1),1));

% States signal
f1_states=f1_cos-1i*f1_sin;

% F1 Fourier transform
spectrum=fftshift(fft(f1_states,parameters.zerofill(1),2),2);

% Plotting
figure(); scale_figure([1.5 2.0]);
plot_2d(spin_system,-real(spectrum),parameters,...
        20,[0.02 0.2 0.02 0.2],2,256,6,'both');





