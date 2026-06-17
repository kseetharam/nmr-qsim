% Minimal Redfield simulation of a 19F-13C-1H three-spin system.
%
% Geometry and CSA tensors are read from the 4-fluorophenylalanine DFT
% calculation in circ_sim/data/operator_dist/4_fluoro_phe.log.
% Nuclei: 19F (DFT index 19), 13C (DFT index 8), 1H (DFT index 11).
% Scalar couplings: 1J(19F-13C) = 243.5 Hz,  1J(13C-1H) = 10.71 Hz.
% No 19F-1H coupling is included.
%
% The function performs the same population tracking and positivity check
% as fc_pair_redfield.m but extended to an 8x8 Hilbert space (64x64
% Liouville space at bas.approximation='none').
%
% Initial state: |uuu> = |up,up,up> for all three spins.
% Eigenstates are classified as S=3/2 (quartet) or S=1/2 (two doublets).
%
%   tau_c   rotational correlation time (s)
%   B0      static field (T); use 0.0 for ZULF
%   approx  Spinach basis approximation ('none', 'IK-2', 'IK-1', 'IK-0')
%
% Example:
%   fch_triple_redfield()
%   fch_triple_redfield(1e-9, 0.0, 'none')
%
% Requires Spinach 2.7 or later.

function [spin_system, R] = fch_triple_redfield(tau_c, B0, approx)

if nargin < 1, tau_c = 1e-5;    end
if nargin < 2, B0    = 0.0;     end
if nargin < 3, approx = 'none'; end

%% Locate the Gaussian log file
script_dir = fileparts(mfilename('fullpath'));
log_file   = fullfile(script_dir, '..', '..', 'data', 'operator_dist', '4_fluoro_phe.log');

%% Extract geometry and CSA from DFT output
[~, inter_dft] = g2spinach(gparse(log_file), ...
    {{'C','13C'}, {'F','19F'}, {'H','1H'}}, ...
    [186.38 192.97 33.44], []);

IDX_F = 19;
IDX_C = 8;
IDX_H = 11;

%% System: spin 1 = 19F, spin 2 = 13C, spin 3 = 1H
sys.isotopes = {'19F', '13C', '1H'};
sys.magnet   = B0;

inter.zeeman.matrix    = cell(1, 3);
inter.zeeman.matrix{1} = inter_dft.zeeman.matrix{IDX_F};
inter.zeeman.matrix{2} = inter_dft.zeeman.matrix{IDX_C};
inter.zeeman.matrix{3} = inter_dft.zeeman.matrix{IDX_H};

inter.coordinates    = cell(3, 1);
inter.coordinates{1} = inter_dft.coordinates{IDX_F};
inter.coordinates{2} = inter_dft.coordinates{IDX_C};
inter.coordinates{3} = inter_dft.coordinates{IDX_H};

inter.coupling.scalar        = cell(3, 3);
inter.coupling.scalar{1, 2}  = 243.5;   % 1J(19F-13C) Hz
inter.coupling.scalar{2, 3}  = 10.71;   % 1J(13C-1H) Hz

inter.relaxation  = {'redfield'};
inter.rlx_keep    = 'labframe';
inter.equilibrium = 'zero';
inter.tau_c       = {tau_c};

bas.formalism     = 'sphten-liouv';
bas.approximation = approx;

spin_system = create(sys, inter);
spin_system = basis(spin_system, bas);

[H, Q] = hamiltonian(assume(spin_system, 'labframe'));  %#ok<ASGLU>
R = relaxation(spin_system);

n_liouv = size(H, 1);
fprintf('Liouville-space dimension: %d\n', n_liouv);

%% R eigenvalue pre-check
eigs_R     = eig(full(R));
re_eigs_R  = sort(real(eigs_R), 'descend');
n_positive = sum(re_eigs_R > 1e-10);

fprintf('R eigenvalues (real parts) — max: %.4f   min: %.4f\n', ...
    re_eigs_R(1), re_eigs_R(end));
if n_positive > 0
    fprintf('  *** %d eigenvalue(s) with positive real part — positivity ALREADY violated ***\n', n_positive);
    fprintf('  Positive real parts: ');
    fprintf('%.4f  ', re_eigs_R(1:n_positive));
    fprintf('\n');
else
    fprintf('  All eigenvalues non-positive — no immediate breakdown from R alone\n');
end

%% Initial state: rho0 = |up,up,up><up,up,up| in sphten-liouv basis
%
% |up><up| = E/2 + Lz  for each spin-1/2.
% Expanding the three-fold product:
%
%   |uuu><uuu| = (E/2+Lz1)(E/2+Lz2)(E/2+Lz3)
%              = (1/8)E@E@E
%              + (1/4)(Lz1@E@E + E@Lz2@E + E@E@Lz3)
%              + (1/2)(Lz1@Lz2@E + Lz1@E@Lz3 + E@Lz2@Lz3)
%              + Lz1@Lz2@Lz3
rho0 = (1/8) * state(spin_system, 'E',                  'all')   ...
     + (1/4) * state(spin_system, 'Lz',                  1)      ...
     + (1/4) * state(spin_system, 'Lz',                  2)      ...
     + (1/4) * state(spin_system, 'Lz',                  3)      ...
     + (1/2) * state(spin_system, {'Lz','Lz'},           {1,2})  ...
     + (1/2) * state(spin_system, {'Lz','Lz'},           {1,3})  ...
     + (1/2) * state(spin_system, {'Lz','Lz'},           {2,3})  ...
     +         state(spin_system, {'Lz','Lz','Lz'},      {1,2,3});

%% Hilbert-space Hamiltonian (8x8) — isotropic Larmor + J-couplings
%
% Built from kron products of explicit 2x2 spin-1/2 matrices (_h suffix),
% independent of Spinach's formalism.
gamma_F = spin('19F');
gamma_C = spin('13C');
gamma_H = spin('1H');
omega_F = -gamma_F * B0;
omega_C = -gamma_C * B0;
omega_H = -gamma_H * B0;

iz = [0.5, 0; 0, -0.5];  ip = [0, 1; 0, 0];  im = [0, 0; 1, 0];  E2 = eye(2);

Lz1_h = kron(kron(iz, E2), E2);
Lz2_h = kron(kron(E2, iz), E2);
Lz3_h = kron(kron(E2, E2), iz);

Lp1_h = kron(kron(ip, E2), E2);  Lm1_h = kron(kron(im, E2), E2);
Lp2_h = kron(kron(E2, ip), E2);  Lm2_h = kron(kron(E2, im), E2);
Lp3_h = kron(kron(E2, E2), ip);  Lm3_h = kron(kron(E2, E2), im);

J_FC = 2*pi * inter.coupling.scalar{1, 2};
J_CH = 2*pi * inter.coupling.scalar{2, 3};

H_hilb = omega_F*Lz1_h + omega_C*Lz2_h + omega_H*Lz3_h ...
       + J_FC*(Lz1_h*Lz2_h + 0.5*(Lp1_h*Lm2_h + Lm1_h*Lp2_h)) ...
       + J_CH*(Lz2_h*Lz3_h + 0.5*(Lp2_h*Lm3_h + Lm2_h*Lp3_h));

[V, D] = eig(H_hilb);
n_hilb = 8;

fprintf('H eigenenergies / (2pi) (Hz): ');
fprintf('%.2f  ', diag(D).' / (2*pi));
fprintf('\n');

%% Classify eigenstates by total spin S and projection M
%
% S^2 = sum_i S_i^2 + 2*sum_{i<j} S_i.S_j
%      = (9/4)*I_8 + 2*(S1.S2 + S1.S3 + S2.S3)
% For S=3/2: S(S+1)=15/4=3.75;  S=1/2: S(S+1)=3/4=0.75.
% Midpoint threshold at 9/4 = 2.25 separates the two families.
% Two S=1/2 doublets share the same (S,M) quantum numbers; they are
% distinguished by the order in which they appear (ascending eigenvalue)
% and labeled with suffix (1) or (2).
dot12 = Lz1_h*Lz2_h + 0.5*(Lp1_h*Lm2_h + Lm1_h*Lp2_h);
dot13 = Lz1_h*Lz3_h + 0.5*(Lp1_h*Lm3_h + Lm1_h*Lp3_h);
dot23 = Lz2_h*Lz3_h + 0.5*(Lp2_h*Lm3_h + Lm2_h*Lp3_h);
S2_mat = (9/4)*eye(n_hilb) + 2*(dot12 + dot13 + dot23);
Sz_mat = Lz1_h + Lz2_h + Lz3_h;

state_labels = cell(1, n_hilb);
seen_half_labels = {};  % track S=1/2 base labels to assign (1)/(2) suffix

for n = 1:n_hilb
    S2_exp = real(V(:,n)' * S2_mat * V(:,n));
    Sz_exp = real(V(:,n)' * Sz_mat * V(:,n));

    % Half-integer M: multiply by 2, round, then display as fraction
    M2 = round(2 * Sz_exp);  % 2*M is an integer
    if M2 > 0
        M_str = sprintf('+%d/2', abs(M2));
    else
        M_str = sprintf('-%d/2', abs(M2));
    end

    if S2_exp > 2.25   % S = 3/2
        state_labels{n} = sprintf('S=3/2, M=%s', M_str);
    else               % S = 1/2
        base = sprintf('S=1/2, M=%s', M_str);
        suffix = sum(strcmp(seen_half_labels, base)) + 1;
        seen_half_labels{end+1} = base;  %#ok<AGROW>
        state_labels{n} = sprintf('S=1/2(%d), M=%s', suffix, M_str);
    end
end

fprintf('Eigenstate labels (ascending energy):');
fprintf('  [%s]', state_labels{:});
fprintf('\n');

%% T matrix: Liouville -> Hilbert conversion  (64 x n_liouv)
%
% Systematically probe all 4^3 = 64 product operators {E,Lz,L+,L-}^3.
% For each triple (a,b,c): build the 8x8 Hilbert matrix via kron, find
% its Spinach sphten-liouv index via max(|state(...)|), and fill one
% column of T so that rho_H = reshape(T * rho_liouv, 8, 8).
ops_1d = {'E', 'Lz', 'L+', 'L-'};
ops_h  = {E2, iz, ip, im};  % corresponding 2x2 matrices

T_liouv2hilb = zeros(n_hilb^2, n_liouv);
%
% py_idx(k) / py_sv(k): Spinach Liouville index and normalisation sv for
% the k-th product operator in Python ordering
%   k = (a-1)*16 + (b-1)*4 + (c-1) + 1  (1-based, a/b/c in 1..4 = E/Lz/L+/L-)
% Key identity used later:  T^{-1} * B_h(:) = sv * e_{idx}
% => R * T^{-1} * B_h(:) = sv * R(:, idx)  (no explicit T inverse needed)
py_idx = zeros(1, n_hilb^2);
py_sv  = zeros(1, n_hilb^2);

for a = 1:4
    for b = 1:4
        for c = 1:4
            k_py   = (a-1)*16 + (b-1)*4 + (c-1) + 1;
            B_h    = kron(kron(ops_h{a}, ops_h{b}), ops_h{c});
            combo  = [a, b, c];
            active = find(combo > 1);  % spins carrying a non-identity operator

            if isempty(active)
                sv = full(state(spin_system, 'E', 'all'));
            elseif numel(active) == 1
                sv = full(state(spin_system, ops_1d{combo(active)}, active));
            else
                op_cell  = ops_1d(combo(active));
                idx_cell = num2cell(active);
                sv = full(state(spin_system, op_cell, idx_cell));
            end

            [sv_max, idx] = max(abs(sv));
            if sv_max > 1e-12
                T_liouv2hilb(:, idx) = B_h(:) / sv(idx);
                py_idx(k_py) = idx;
                py_sv(k_py)  = sv(idx);
            end
        end
    end
end

liouv2hilb = @(v) reshape(T_liouv2hilb * full(v), n_hilb, n_hilb);

%% Liouville-space projectors for eigenstate populations
%
% P_n(t) = Tr(|n><n| * rho_H(t))
%        = (T' * vec(|n><n|))' * rho_liouv(t)
%
% T' (transpose), not T^{-1}: equality holds only for orthonormal T, but
% Spinach's product operator basis is NOT Frobenius-orthonormal (e.g.
% ||E_8||_F = 2*sqrt(2)), so T' is the correct projector here.
proj_liouv = zeros(n_liouv, n_hilb);
for n = 1:n_hilb
    rho_n = V(:,n) * V(:,n)';
    proj_liouv(:, n) = T_liouv2hilb' * rho_n(:);
end

%% Transition rates from Spinach R:  d<P_u>/dt|_{t=0} = <u| R |rho_0>
%
% For any eigenstate projector u = |n><n|, the initial rate of population
% transfer into u from rho_0 = |3/2,+3/2><3/2,+3/2| equals
%
%   rate_u = Re( proj_liouv(:,u)' * R * rho0 )
%
% This is valid because rho_0 is already an eigenstate of H0, so
% -i[H, rho_0] = 0 and the Hamiltonian part of the generator contributes
% nothing to population changes at t=0.

evals_hilb = diag(D);   % 8x1 ascending eigenvalues (rad/s), same as diag(D)

% Locate target eigenstates by label
n_p32 = find(strcmp(state_labels, 'S=3/2, M=+3/2'), 1);   % initial state
n_q12 = find(strcmp(state_labels, 'S=3/2, M=+1/2'), 1);   % Case 1
n_m32 = find(strcmp(state_labels, 'S=3/2, M=-3/2'), 1);   % sanity check

% |1/2,+1/2> doublet closer in energy to the quartet
E_quartet    = evals_hilb(n_p32);
half12_idx   = find(cellfun(@(s) ~isempty(regexp(s, 'S=1/2\(\d\), M=\+1/2')), state_labels));
[~, ord]     = sort(abs(evals_hilb(half12_idx) - E_quartet));
n_d12_close  = half12_idx(ord(1));

% Compute R * rho0 once (full matrix-vector product in Liouville space)
R_rho0 = full(R) * full(rho0);

fprintf('\n%s\n', repmat('=', 1, 65));
fprintf('Spinach Redfield rates:  d<P_u>/dt|_{t=0} = <u| R |rho_0>\n');
fprintf('  Initial state = |3/2,+3/2>  (eigenstate %d,  E = %.4f Hz)\n', ...
    n_p32, evals_hilb(n_p32) / (2*pi));
fprintf('%s\n', repmat('=', 1, 65));

rate_descs = { ...
    n_p32, '|3/2,+3/2>  (initial — expect negative out-rate)'; ...
    n_q12, '|3/2,+1/2>  (Case 1 — intra-quartet DeltaM=-1)'; ...
    n_d12_close, sprintf('|1/2,+1/2> close  [E = %.2f Hz]  (Case 2 — inter-manifold)', ...
                         evals_hilb(n_d12_close)/(2*pi)); ...
    n_m32, '|3/2,-3/2>  (sanity check — DeltaM=-3, expect ~0)' };

for k = 1:size(rate_descs, 1)
    n_u  = rate_descs{k, 1};
    desc = rate_descs{k, 2};
    rate = real(proj_liouv(:, n_u)' * R_rho0);
    fprintf('  Target: %s\n', desc);
    fprintf('    Label     : ''%s''\n', state_labels{n_u});
    fprintf('    Rate (rad/s) : %+.6e\n', rate);
    fprintf('    Rate (Hz)    : %+.6e\n\n', rate / (2*pi));
end

%% R matrix in Python product-operator basis
%
% Computes R_Spinach_prod[i,j] = Tr{ B_i' * R_superop[B_j] }
% in the same {E, Lz, L+, L-}^3 ordering used by compute_R_lindblad.py,
% enabling direct element-wise comparison with R_L.
%
% Derivation (no T inverse needed):
%   T * (sv_j * e_{idx_j}) = B_j(:)          [from T matrix construction]
%   => T^{-1} * B_j(:)     = sv_j * e_{idx_j}
%   => R * T^{-1} * B_j(:) = sv_j * R(:, idx_j)
%   => R_superop[B_j] in Hilbert space = reshape(T * sv_j * R(:,idx_j), 8,8)
%   => R_Spinach_prod(i,j)  = trace(B_i' * that)

n_py   = n_hilb^2;   % 64
R_full = full(R);
R_Spinach_prod = zeros(n_py, n_py);

% Reconstruct Python-ordered operator labels in MATLAB
ops_lbl = {'E', 'Lz', 'L+', 'L-'};
py_labels = cell(1, n_py);
for k = 1:n_py
    ai = floor((k-1)/16) + 1;
    bi = floor(mod(k-1, 16)/4) + 1;
    ci = mod(k-1, 4) + 1;
    py_labels{k} = sprintf('%sx%sx%s', ops_lbl{ai}, ops_lbl{bi}, ops_lbl{ci});
end

for j_py = 1:n_py
    idx_j = py_idx(j_py);
    sv_j  = py_sv(j_py);
    if idx_j == 0, continue; end

    % R applied to j-th product operator, result in Hilbert space
    R_Bj_hilb = reshape(T_liouv2hilb * (sv_j * R_full(:, idx_j)), n_hilb, n_hilb);

    for i_py = 1:n_py
        ai = floor((i_py-1)/16) + 1;
        bi = floor(mod(i_py-1, 16)/4) + 1;
        ci = mod(i_py-1, 4) + 1;
        B_i = kron(kron(ops_h{ai}, ops_h{bi}), ops_h{ci});
        R_Spinach_prod(i_py, j_py) = trace(B_i' * R_Bj_hilb);
    end
end

% Gram diagonal: G(k) = Tr(B_k' * B_k) = ||B_k||_HS^2
% Used to normalise matrix elements: R_norm(k,k) = R_raw(k,k) / G(k)
gram_diag = zeros(n_py, 1);
for k = 1:n_py
    ai = floor((k-1)/16) + 1;
    bi = floor(mod(k-1, 16)/4) + 1;
    ci = mod(k-1, 4) + 1;
    B_k = kron(kron(ops_h{ai}, ops_h{bi}), ops_h{ci});
    gram_diag(k) = real(trace(B_k' * B_k));
end

% Save for external use
data_dir = fullfile(script_dir, 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
save(fullfile(data_dir, 'R_Spinach_prod.mat'), ...
    'R_Spinach_prod', 'py_labels', 'py_idx', 'py_sv', 'gram_diag');
fprintf('R_Spinach_prod saved -> %s\n', fullfile(data_dir, 'R_Spinach_prod.mat'));

% Load Python R_L and compare
R_L_path = fullfile(data_dir, 'R_L_lindblad_gamma1.mat');
if exist(R_L_path, 'file')
    d   = load(R_L_path);
    R_L = d.R_L;

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('R_Spinach_prod  vs  R_L (Python Lindblad, gamma_scale = 1.0)\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('  max |Re(R_Spinach_prod)| = %.6e rad/s\n', max(abs(real(R_Spinach_prod(:)))));
    fprintf('  max |Re(R_L)|            = %.6e rad/s\n', max(abs(real(R_L(:)))));
    fprintf('  max |R_Spinach - R_L|    = %.6e rad/s\n', max(abs(R_Spinach_prod(:) - R_L(:))));
    fprintf('  max |R_Spinach - 2*R_L|  = %.6e rad/s\n', max(abs(R_Spinach_prod(:) - 2*R_L(:))));

    fprintf('\n  Diagonal R[k,k] comparison  (non-zero entries, Hz):\n');
    fprintf('  Columns marked (*) are normalised by G(k) = Tr(B_k^dagger B_k)\n');
    fprintf('  %3s  %-22s  %6s  %16s  %16s  %16s  %16s  %8s\n', ...
        'k', 'operator', 'G(k)', ...
        'R_Spinach/2pi', 'R_Spinach*/2pi', ...
        'R_L/2pi',       'R_L*/2pi', ...
        'ratio*');
    fprintf('  %s\n', repmat('-', 1, 100));
    for k = 1:n_py
        rs_raw  = real(R_Spinach_prod(k,k)) / (2*pi);
        rl_raw  = real(R_L(k,k))            / (2*pi);
        G       = gram_diag(k);
        rs_norm = rs_raw / G;
        rl_norm = rl_raw / G;
        ratio   = rs_norm / rl_norm;
        if abs(rl_raw) > 1e-4
            fprintf('  %3d  %-22s  %6.2f  %16.6e  %16.6e  %16.6e  %16.6e  %8.4f\n', ...
                k-1, py_labels{k}, G, rs_raw, rs_norm, rl_raw, rl_norm, ratio);
        end
    end
else
    fprintf('  R_L file not found at %s\n  Run compute_R_lindblad.py first.\n', R_L_path);
end

%% Time grid: span 10 correlation times
t_end   = 10 * tau_c;
n_steps = 500;
t_axis  = linspace(0, t_end, n_steps);
dt      = t_axis(2) - t_axis(1);

%% Propagation under Redfield generator G = -i*H + R
G      = -1i*H + R;
P_step = expm(full(G) * dt);
rho_t  = full(rho0);

populations = zeros(n_hilb, n_steps);
min_eig_rho = zeros(1, n_steps);

for k = 1:n_steps
    for n = 1:n_hilb
        populations(n, k) = real(proj_liouv(:,n)' * rho_t);
    end
    rho_H = liouv2hilb(rho_t);
    min_eig_rho(k) = min(real(eig(rho_H)));
    rho_t = P_step * rho_t;
end

%% Plot
if t_end < 1e-6
    t_scale = 1e9;  t_unit = 'ns';
elseif t_end < 1e-3
    t_scale = 1e6;  t_unit = '\mus';
else
    t_scale = 1e3;  t_unit = 'ms';
end

% Style LUT: {label, RGB, linestyle, marker, linewidth}
% Blue family  → S=3/2 quartet;  orange/red → S=1/2 doublet (1);
% green/teal   → S=1/2 doublet (2).
style_lut = {
    'S=3/2, M=+3/2',    [0.00, 0.45, 0.74], '-',   'none', 2.0;
    'S=3/2, M=+1/2',    [0.30, 0.65, 0.94], '--',  'none', 1.8;
    'S=3/2, M=-1/2',    [0.00, 0.25, 0.54], ':',   's',    2.0;
    'S=3/2, M=-3/2',    [0.10, 0.10, 0.80], '-.',  '^',    1.8;
    'S=1/2(1), M=+1/2', [0.85, 0.33, 0.10], '-',   'none', 2.0;
    'S=1/2(1), M=-1/2', [0.93, 0.69, 0.13], '--',  'd',    1.8;
    'S=1/2(2), M=+1/2', [0.47, 0.67, 0.19], '-',   'none', 2.0;
    'S=1/2(2), M=-1/2', [0.10, 0.60, 0.50], '--',  'o',    1.8;
};
mk_idx = round(linspace(1, n_steps, 20));

fig = figure('Position', [100 100 960 560]);

subplot(2,1,1);
hold on;
for n = 1:n_hilb
    row = find(strcmp(style_lut(:,1), state_labels{n}));
    if isempty(row), row = mod(n-1, size(style_lut,1)) + 1; end  % fallback
    clr = style_lut{row, 2};
    ls  = style_lut{row, 3};
    mk  = style_lut{row, 4};
    lw  = style_lut{row, 5};
    if strcmp(mk, 'none')
        plot(t_axis*t_scale, populations(n,:), ...
            'Color', clr, 'LineStyle', ls, 'LineWidth', lw, ...
            'DisplayName', state_labels{n});
    else
        plot(t_axis*t_scale, populations(n,:), ...
            'Color', clr, 'LineStyle', ls, 'LineWidth', lw, ...
            'Marker', mk, 'MarkerSize', 5, 'MarkerIndices', mk_idx, ...
            'MarkerFaceColor', clr, 'DisplayName', state_labels{n});
    end
end
hold off;
xlabel(['Time (' t_unit ')'], 'FontSize', 13);
ylabel('Population', 'FontSize', 13);
legend('FontSize', 10, 'Location', 'best');
title(sprintf(['^{19}F-^{13}C-^{1}H | \\tau_c = %.1e s | ' ...
    'B_0 = %.2f T | approx: %s'], tau_c, B0, approx), ...
    'FontSize', 12, 'Interpreter', 'tex');
grid on; box on; set(gca, 'FontSize', 11);

subplot(2,1,2);
plot(t_axis*t_scale, min_eig_rho, 'k', 'LineWidth', 1.5);
yline(0, 'r--', 'LineWidth', 1.2);
xlabel(['Time (' t_unit ')'], 'FontSize', 13);
ylabel('min eig(\rho)', 'FontSize', 13);
title('Positivity check — breakdown when < 0', 'FontSize', 12);
grid on; box on; set(gca, 'FontSize', 11);

set(fig, 'PaperPositionMode', 'auto');

%% Save figure as PDF
pdf_name = fullfile(script_dir, ...
    sprintf('fch_triple_tc%.0e_B%.2f_%s_refJs.pdf', tau_c, B0, approx));
if exist('exportgraphics', 'file')
    exportgraphics(fig, pdf_name, 'ContentType', 'vector');
else
    print(fig, pdf_name, '-dpdf', '-bestfit');
end
fprintf('Figure saved: %s\n', pdf_name);

end
