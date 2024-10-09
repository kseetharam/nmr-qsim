clear;

datapath = path_fix();
addpath(genpath('/n/holyscratch01/jaffe_lab/Everyone/kis/nmr-qsim/spinach_examples/liquids/'))
addpath(genpath('/n/holyscratch01/jaffe_lab/Everyone/kis/nmr-qsim/spinach_examples/solids/'))
load(strcat(datapath,'generators_noesyhsqc_ubiquitin_deut.mat'))

% datapath = '/Users/kis/KIS Dropbox/Kushal Seetharam/NMR QSim/Code/data/';
% load(strcat(datapath,'generator_data/liquids/generators_noesyhsqc_ubiquitin_deut.mat'))

M = size(H,1);  % linear dimension of reduced basis (H and R are square matrices)

H = inflate(H);
R = inflate(R);

H_density = num2str(100*nnz(H)/numel(H));  % percentage of non-zero elements
R_density = num2str(100*nnz(R)/numel(R));  % percentage of non-zero elements

%% Compute sparsity

parpool('local', str2num(getenv('SLURM_CPUS_PER_TASK')))

tic

H_row = zeros(M,1); H_col = zeros(M,1);
R_row = zeros(M,1); R_col = zeros(M,1);

parfor i=1:M
    H_row(i) = nnz(H(i,:)); H_col(i) = nnz(H(:,i));
    R_row(i) = nnz(R(i,:)); R_col(i) = nnz(R(:,i));
end

H_row_sparsity = max(H_row); H_col_sparsity = max(H_col);
R_row_sparsity = max(R_row); R_col_sparsity = max(R_col);

toc

disp('H_row_sparsity')
disp(H_row_sparsity)
disp('H_col_sparsity')
disp(H_col_sparsity)
disp('R_row_sparsity')
disp(R_row_sparsity)
disp('R_col_sparsity')
disp(R_col_sparsity)

delete(gcp);
exit;