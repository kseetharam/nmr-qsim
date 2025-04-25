clc
clear all
close all

% Magnet field
sys.magnet=18.78;

% Isotopes
sys.isotopes={'1H','1H','1H','1H'};

% Chemical shifts
inter.zeeman.scalar={3.69 1.39 1.39 1.39};

% J-couplings
inter.coupling.scalar=cell(4,4);
%inter.coupling.scalar{1,2}=0.0;
%inter.coupling.scalar{1,3}=0.0;
%inter.coupling.scalar{1,4}=0.0;
inter.coupling.scalar{1,2}=7.0;
inter.coupling.scalar{1,3}=7.0;
inter.coupling.scalar{1,4}=7.0;

% Cartesian coordinates 
inter.coordinates={[0.6861    0.2705    1.5010];
                   [1.3077    1.1298   -1.3993];
                   [0.7905    2.2125   -0.0860];
                   [2.3693    1.3798    0.0233]};      

% Relaxation theory parameters
inter.relaxation={'redfield'};
inter.equilibrium='dibari';
inter.temperature=298;
inter.rlx_keep='kite';
inter.tau_c={18.8e-12};

% inter.r1_rates={0.19 0.45 0.45 0.45};
% inter.r2_rates={30 20 20 20};
% inter.lind_r1_rates=[0.19 0.45 0.45 0.45];
% inter.lind_r2_rates=[18 10 10 10];

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='none';

% Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

% Experiment parameters
parameters.tmix=800e-3;
parameters.offset=3759.37;
parameters.sweep=[6398.95 9615.38];
parameters.npoints = [2048 2048]/2;
parameters.zerofill = [2048 2048];
%parameters.npoints = [2048 2048];
%parameters.zerofill = 2*[2048 2048];
%parameters.npoints = [10240 10240];
%parameters.zerofill = [10240 10240];
%parameters.npoints=[2048 2048];
%parameters.zerofill=2*[2048 2048];
parameters.spins={'1H'};
parameters.needs={'rho_eq'};

% NOESY simulation
[fid,H,R,rho0]=liquid_mod(spin_system,@noesy,parameters,'nmr');



% Apodization


%{

fid.cos=apodisation(spin_system,fid.cos,{{'gauss',21},{'gauss',21}}); 
fid.sin=apodisation(spin_system,fid.sin,{{'gauss',21},{'gauss',21}});  

% F2 Fourier transform
f1_cos=real(fftshift(fft(fid.cos,parameters.zerofill(2),1),1));
f1_sin=real(fftshift(fft(fid.sin,parameters.zerofill(2),1),1));

% States signal
f1_states=f1_cos-1i*f1_sin;

% F1 Fourier transform
spectrum=fftshift(fft(f1_states,parameters.zerofill(1),2),2);


% Plotting

figure();
scale_figure([1.5 2.0]);
[axis_f1,axis_f2,spectrum_out] = plot_2d(spin_system,-real(spectrum),parameters,...
        20,[0.02 0.2 0.02 0.2],2,256,6,'both');
%}

% spec_cross = spectrum_out(1520:1540,914:926);
% spec_diag = spectrum_out(1520:1540,1698:1710);
% peak_cross = spec_cross(abs(spec_cross) == max(abs(spec_cross(:))));
% peak_diag = max(max(spec_diag));
% peak_cross/peak_diag

%% Brute force
%{
dt=2./parameters.sweep;
Lx=operator(spin_system,'Lx',parameters.spins{1});
Ly=operator(spin_system,'Ly',parameters.spins{1});
Lz=operator(spin_system,'Lz',parameters.spins{1});

coil=state(spin_system,'L+',parameters.spins{1},'cheap');

% first 90x pulse
U90x = expm(-1i*Lx*pi/2);
rho_initial = U90x*rho0;
dim = length(rho0);
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

%}

load('../circ_sim/ALA_truncR_FIDcos_Grad_ZZNoEffJump.mat')
load('../circ_sim/ALA_truncR_FIDsin_Grad_ZZNoEffJump.mat')
fid_brute.cos = FID_cos;
fid_brute.sin = FID_sin;


% Apodization
fid_brute.cos=apodisation(spin_system,fid_brute.cos,{{'gauss',21},{'gauss',21}}); 
fid_brute.sin=apodisation(spin_system,fid_brute.sin,{{'gauss',21},{'gauss',21}});  


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

%fid,H,R,rho0


%p = parameters;

%p.Tpts = parameters.npoints(1);
%p.dt = dt(1); %the delta of time to do the sampling of points...
%p.H = H;
%p.R = R;
%p.fid = fid;
%p.rho0 = rho0;
%p.coil = coil;
%p.Lx = Lx;
%p.Ly = Ly;
%p.Lz = Lz;
% p.fid_test = fid_test; % the fid's obtained via explicit matrix
% exponentiation

%save NOESYdata_ALA_experiment_HNoJs.mat p

