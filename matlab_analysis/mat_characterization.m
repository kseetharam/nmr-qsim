
addpath('spinach_examples/liquids/')
addpath('spinach_examples/solids/')

% H = H1; R = R1;
% H = H2; R = R2;

M = size(H,1);  % linear dimension of reduced basis (H and R are square matrices)

H = inflate(H);
R = inflate(R);

H_density = num2str(100*nnz(H)/numel(H));  % percentage of non-zero elements
R_density = num2str(100*nnz(R)/numel(R));  % percentage of non-zero elements

%% Compute eigenvalues

% datapath = '../data/generator_data/liquids';
% 
% % % Full matrix
% % lambda_H = eig(full(H));
% % lambda_R = eig(full(R));
% % lambda_L = eig(full(H+R));
% % save(strcat(datapath,'/eigenvalues_noesy_strychnine.mat'), 'lambda_H', 'lambda_R', 'lambda_L', '-v7.3')
% 
% M = 1
% % Sparse matrix
% lambda_H = eigs(H,M);
% lambda_R = eigs(R,M);
% lambda_L = eigs(H+R,M);
% % save(strcat(datapath,'/eigenvalues_sparse_noesy_ubiquitin.mat'), 'lambda_H', 'lambda_R', 'lambda_L', '-v7.3')

%%

% figure(1)
% hold on
% plot(lambda_H,'k')
% plot(lambda_R,'r')
% plot(lambda_L,'g')
% legend()

%% Compute sparsity

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

%% Visualize

% figure(1)
% spy(H)
% figure(2)
% spy(R)