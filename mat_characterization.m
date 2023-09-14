
addpath('spinach_examples/liquids/')
addpath('spinach_examples/solids/')

% H = H1; R = R1;
% H = H2; R = R2;

M = size(H,1);  % linear dimension of reduced basis (H and R are square matrices)

H = inflate(H);
R = inflate(R);

%% Compute eigenvalues

datapath = '../data/generator_data/liquids';

% % Full matrix
% lambda_H = eig(full(H));
% lambda_R = eig(full(R));
% lambda_L = eig(full(H+R));
% save(strcat(datapath,'/eigenvalues_noesy_strychnine.mat'), 'lambda_H', 'lambda_R', 'lambda_L', '-v7.3')

M = 1
% Sparse matrix
lambda_H = eigs(H,M);
lambda_R = eigs(R,M);
lambda_L = eigs(H+R,M);
% save(strcat(datapath,'/eigenvalues_sparse_noesy_ubiquitin.mat'), 'lambda_H', 'lambda_R', 'lambda_L', '-v7.3')

%%

figure(1)
hold on
plot(lambda_H,'k')
plot(lambda_R,'r')
plot(lambda_L,'g')
legend()

%% Compute row sparsity

H_row_sparsity = -1;
R_row_sparsity = -1;
for i=1:M
    H_temp = nnz(H(i,:));
    if H_temp > H_row_sparsity
        H_row_sparsity = H_temp;
    end
    R_temp = nnz(R(i,:));
    if R_temp > R_row_sparsity
        R_row_sparsity = R_temp;
    end
end

%% Compute column sparsity

H_col_sparsity = -1;
R_col_sparsity = -1;
for i=1:M
    H_temp = nnz(H(:,i));
    if H_temp > H_col_sparsity
        H_col_sparsity = H_temp;
    end
    R_temp = nnz(R(:,i));
    if R_temp > R_col_sparsity
        R_col_sparsity = R_temp;
    end
end

%% Visualize

% figure(1)
% spy(H)
% figure(2)
% spy(R)