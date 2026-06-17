% Minimal Redfield simulation of a 19F-13C two-spin pair — zeeman-hilb formalism.
%
% Produces the same population tracking and positivity check as
% fc_pair_redfield.m (sphten-liouv version) but uses Spinach's zeeman-hilb
% formalism, where the density matrix lives directly in Hilbert space as a
% 4x4 matrix.  There is no Liouville<->Hilbert basis conversion: everything
% is explicit.
%
%   hamiltonian()  ->  4x4 Hilbert-space Hamiltonian H
%   relaxation()   ->  16x16 Redfield superoperator R acting on rho(:)
%                      (standard MATLAB column-major vectorization)
%
% The Liouville generator is assembled as:
%   G = -i*(kron(I,H) - kron(H.',I)) + R        (16x16)
%
% Populations are tracked in the eigenbasis of H at the given B0.
% At B0=0 the eigenstates approach the singlet/triplet basis.
%
%   tau_c   rotational correlation time (s), e.g. 1e-9
%   B0      static field (T); use 0.0 for ZULF
%
% Example:
%   fc_pair_redfield_zhlb(1e-9, 0.0)
%
% Requires Spinach 2.7 or later.

function fc_pair_redfield_zhlb(tau_c, B0)

if nargin < 1, tau_c = 1e-9; end
if nargin < 2, B0    = 0.0;  end

%% Gaussian log file
script_dir = fileparts(mfilename('fullpath'));
log_file   = fullfile(script_dir, '..', '..', 'data', 'operator_dist', '4_fluoro_phe.log');

%% DFT geometry and CSA
[~, inter_dft] = g2spinach(gparse(log_file), ...
    {{'C','13C'}, {'F','19F'}, {'H','1H'}}, ...
    [186.38 192.97 33.44], []);

IDX_C = 8;
IDX_F = 19;

%% Spin system
sys.isotopes = {'19F', '13C'};
sys.magnet   = B0;

inter.zeeman.matrix    = cell(1, 2);
inter.zeeman.matrix{1} = inter_dft.zeeman.matrix{IDX_F};
inter.zeeman.matrix{2} = inter_dft.zeeman.matrix{IDX_C};

inter.coordinates    = cell(2, 1);
inter.coordinates{1} = inter_dft.coordinates{IDX_F};
inter.coordinates{2} = inter_dft.coordinates{IDX_C};

inter.coupling.scalar       = cell(2, 2);
inter.coupling.scalar{1, 2} = 243.5;   % 1J(19F-13C) in Hz

inter.relaxation  = {'redfield'};
inter.rlx_keep    = 'labframe';
inter.equilibrium = 'zero';
inter.tau_c       = {tau_c};

%% Basis — Zeeman Hilbert space (no Liouville-space truncation)
bas.formalism = 'zeeman-hilb';
bas.approximation = 'none';

%% Spinach housekeeping
spin_system = create(sys, inter);
spin_system = basis(spin_system, bas);

[H, Q] = hamiltonian(assume(spin_system, 'labframe'));  %#ok<ASGLU>
R = relaxation(spin_system);

H = full(H);   % 4x4 Hilbert-space Hamiltonian
R = full(R);   % 16x16 Redfield superoperator on vec(rho)

fprintf('H: %dx%d    R: %dx%d\n', size(H,1), size(H,2), size(R,1), size(R,2));

%% Eigenvalue analysis of R
eigs_R     = eig(R);
re_eigs_R  = sort(real(eigs_R), 'descend');
n_positive = sum(re_eigs_R > 1e-10);

fprintf('R eigenvalues (real parts) — max: %.4f   min: %.4f\n', ...
    re_eigs_R(1), re_eigs_R(end));
if n_positive > 0
    fprintf('  *** %d eigenvalue(s) with positive real part — positivity ALREADY violated ***\n', ...
        n_positive);
    fprintf('  Positive real parts: ');
    fprintf('%.4f  ', re_eigs_R(1:n_positive));
    fprintf('\n');
else
    fprintf('  All eigenvalues non-positive — no immediate breakdown from R alone\n');
end

%% Initial state: |up,down><up,down| as a 4x4 density matrix
%
% Zeeman basis order for sys.isotopes = {'19F','13C'}:
%   index 1: |up_F, up_C>
%   index 2: |up_F, down_C>   <-- this is |up,down>
%   index 3: |down_F, up_C>
%   index 4: |down_F, down_C>
rho0 = zeros(4);
rho0(2, 2) = 1;

%% Eigenbasis of H (4x4) for population tracking
%
% H is taken directly from Spinach, which includes the isotropic Larmor
% frequencies, J-coupling, and CSA contributions from the DFT calculation.
% eig() sorts eigenvalues in ascending order; columns of V are eigenstates.
[V, D] = eig(H);
fprintf('H eigenenergies / (2pi) (Hz): ');
fprintf('%.2f  ', diag(D).' / (2*pi));
fprintf('\n');

%% Liouville generator: G = -i*(I⊗H - H^T⊗I) + R
%
% For d/dt rho = -i[H, rho] + R(rho), vectorized column-major as d/dt vec(rho):
%   vec(-i[H,rho]) = -i*(kron(I,H) - kron(H.',I)) * vec(rho)
n_hilb = size(H, 1);   % 4
L_H = kron(eye(n_hilb), H) - kron(H.', eye(n_hilb));   % commutator superop
G   = -1i * L_H + R;

%% Time grid
J_Hz    = 243.5;
t_end   = 5 / J_Hz;
n_steps = 500;
t_axis  = linspace(0, t_end, n_steps);
dt      = t_axis(2) - t_axis(1);

%% Propagation
P       = expm(G * dt);
rho_vec = rho0(:);   % column-major vectorization of the initial 4x4 matrix

n_eig       = n_hilb;
populations = zeros(n_eig, n_steps);
min_eig_rho = zeros(1, n_steps);

for k = 1:n_steps
    rho_H = reshape(rho_vec, n_hilb, n_hilb);

    % Population of eigenstate |n>: P_n = <n|rho|n> = V(:,n)' * rho_H * V(:,n)
    for n = 1:n_eig
        populations(n, k) = real(V(:,n)' * rho_H * V(:,n));
    end

    % Positivity: smallest eigenvalue of rho_H (no conversion needed)
    min_eig_rho(k) = min(real(eig(rho_H)));

    rho_vec = P * rho_vec;
end

%% Plot
fig = figure('Position', [100 100 960 520]);

subplot(2,1,1);
plot(t_axis * 1e3, populations.', 'LineWidth', 1.5);
xlabel('Time (ms)', 'FontSize', 13);
ylabel('Population', 'FontSize', 13);
legend({'|1\rangle','|2\rangle','|3\rangle','|4\rangle'}, ...
    'FontSize', 11, 'Location', 'best');
title(sprintf(['^{19}F-^{13}C (zeeman-hilb) | \\tau_c = %.1e s | ' ...
    'B_0 = %.2f T'], tau_c, B0), 'FontSize', 12, 'Interpreter', 'tex');
grid on; box on; set(gca, 'FontSize', 11);

subplot(2,1,2);
plot(t_axis * 1e3, min_eig_rho, 'k', 'LineWidth', 1.5);
yline(0, 'r--', 'LineWidth', 1.2);
xlabel('Time (ms)', 'FontSize', 13);
ylabel('min eig(\rho)', 'FontSize', 13);
title('Positivity check — breakdown when < 0', 'FontSize', 12);
grid on; box on; set(gca, 'FontSize', 11);

set(fig, 'PaperPositionMode', 'auto');

end
