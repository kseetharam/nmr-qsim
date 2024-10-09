
clear;

addpath('spinach_examples/liquids/')
addpath('spinach_examples/solids/')
datapath = '../data/generator_data/liquids';

%% Choose molecule

molecule = 'methanol';
% molecule = 'sucrose';
% molecule = 'strychnine';
% molecule = 'ubiquitin';

%% Load data

load(strcat(datapath,'/generators_noesy_',molecule,'.mat'))
% load(strcat(datapath,'/generators_noesyhsqc_ubiquitin_deut.mat'))
% load(strcat(datapath,'/eigenvalues_noesy_',molecule,'.mat'))

M = size(H,1);  % linear dimension of reduced basis (H and R are square matrices)
H = inflate(H);
R = inflate(R);
% L = H+1i*R; % Spinach convention (assuming the propagater e^{-1i*L})
L = -1i*H+R;
Lr = (L+L')/2;

stab_rank_L = (norm(L,'fro')/normest(L))^2; stab_rank_H = (norm(H,'fro')/normest(H))^2; stab_rank_R = (norm(R,'fro')/normest(R))^2;
% stab_rank_L = (norm(L,'fro')/norm(full(L)))^2; stab_rank_H = (norm(H,'fro')/norm(full(H)))^2; stab_rank_R = (norm(R,'fro')/norm(full(R)))^2;

%% Compute eigenvalues

% [V_H,lambda_H] = eig(full(-1i*H));
% [V_R,lambda_R] = eig(full(R));
% [V_L,lambda_L] = eig(full(L));
% [V_Lr,lambda_Lr] = eig(full(Lr));
% 
% iVH = imag(V_H);
% iVR = imag(V_R); isreal(V_R)
% iVL = imag(V_L);

%% Compute metrics

% matDiff = R-Lr;
% lambdaDiff = sum(abs(sort(lambda_R)-sort(lambda_Lr)));
% alpha = sort(real(lambda_L)); alpha = alpha(end:-1:end-1);
% mu = sort(real(lambda_Lr)); mu = mu(end:-1:end-1);
% norm_expL = norm(expm(L));
% 
% save(strcat(datapath,'/metrics_noesy_',molecule,'.mat'), 'lambda_H','V_H','lambda_R','V_R','lambda_L','V_L','alpha','mu','norm_expL', '-v7.3')

%% Plot eigenvalues

% figure(1)
% hold on
% plot(real(lambda_H),imag(lambda_H),'gx')
% plot(real(lambda_R),imag(lambda_R),'rx')
% plot(real(lambda_L),imag(lambda_L),'kx')
% plot(real(lambda_Lr),imag(lambda_Lr),'bx')
% xlabel('Real part')
% ylabel('Imaginary part')
% title('Eigenvalues')
% legend('-iH','R','L','Lr')

%% Visualize

% figure(1)
% spy(H)
% figure(2)
% spy(R)

%% State norm

% [spin_system, parameters, H, R, K, rhot, rho0, obs_p, obs_z] = eval(strcat('dyn_noesy_',molecule));
% timestep_F = 1./parameters.sweep; timestep_mix = parameters.tmix./parameters.npoints_mix;
% tgrid_F1 = timestep_F(1)*[0:parameters.npoints(1)-1]; tgrid_mix = tgrid_F1(end) + timestep_mix*[0:parameters.npoints_mix-1]; tgrid_F2 = tgrid_mix(end) + timestep_F(2)*[0:parameters.npoints(2)-1];
% tgrid = [tgrid_F1, tgrid_mix, tgrid_F2];
% 
% norm_rho0_1 = vecnorm(rho0,1);
% norm_rho0_2 = vecnorm(rho0,2);
% norm_rhot_1 = vecnorm(rhot,1);
% norm_rhot_2 = vecnorm(rhot,2);
% 
% figure(1)
% plot(tgrid,real(obs_p))
% figure(2)
% plot(tgrid,imag(obs_p))
% figure(3)
% plot(tgrid,real(obs_z))
% figure(4)
% plot(tgrid,imag(obs_z))
% figure(5)
% plot(tgrid,1-norm_rhot_1)
% figure(6)
% plot(tgrid,1-norm_rhot_2)