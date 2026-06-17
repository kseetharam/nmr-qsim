% Minimal Redfield simulation of a 19F-13C two-spin pair.
%
% Geometry and CSA tensors are read from the 4-fluorophenylalanine DFT
% calculation in circ_sim/data/operator_dist/4_fluoro_phe.log. Only the
% directly bonded 19F and 13C are included; all other spins are dropped.
%
% The function signature exposes the three parameters most relevant to
% studying Redfield breakdown:
%
%   tau_c        rotational correlation time (s), e.g. 1e-9
%   B0           static field (T), e.g. 18.8 for 800 MHz 1H; use ~0 for ZULF
%   approx       Spinach basis approximation:
%                  'none'  -- exact Redfield (full Liouville space, 16 states)
%                  'IK-2'  -- mild truncation
%                  'IK-1'  -- moderate truncation
%                  'IK-0'  -- secular approximation (most drastic)
%
% Populations are tracked in the eigenbasis of the full Hamiltonian
% (isotropic Larmor + J-coupling) at the given B0.  At ZULF this approaches
% the singlet/triplet basis; at high field it approaches the Zeeman basis.
% A positivity check flags Redfield breakdown (min eigenvalue of rho < 0).
%
%   init_state   initial pure state: 'ud' = |up,down> (default)
%                                   'uu' = |up,up>
%
% Example: run with defaults
%   fc_pair_redfield()
%
% Example: sweep correlation time
%   for tc = [1e-10 1e-9 1e-8 1e-7]
%       fc_pair_redfield(tc, 0.0, 'none');
%   end
%
% Requires Spinach 2.7 or later.

function fc_pair_redfield(tau_c, B0, approx, init_state)

if nargin < 1, tau_c      = 1e-5;    end
if nargin < 2, B0         = 0.0;     end
if nargin < 3, approx     = 'none';  end
if nargin < 4, init_state = 'uu';    end

%% Locate the Gaussian log file relative to this script
script_dir = fileparts(mfilename('fullpath'));
log_file   = fullfile(script_dir, '..', '..', 'data', 'operator_dist', '4_fluoro_phe.log');

%% Extract geometry and CSA from DFT output
[~, inter_dft] = g2spinach(gparse(log_file), ...
    {{'C','13C'}, {'F','19F'}, {'H','1H'}}, ...
    [186.38 192.97 33.44], []);

% Indices in the DFT output corresponding to the para-CF pair
IDX_C = 8;
IDX_F = 19;

%% System: spin 1 = 19F, spin 2 = 13C
sys.isotopes = {'19F', '13C'};
sys.magnet   = B0;

inter.zeeman.matrix    = cell(1, 2);
inter.zeeman.matrix{1} = inter_dft.zeeman.matrix{IDX_F};
inter.zeeman.matrix{2} = inter_dft.zeeman.matrix{IDX_C};

inter.coordinates    = cell(2, 1);
inter.coordinates{1} = inter_dft.coordinates{IDX_F};
inter.coordinates{2} = inter_dft.coordinates{IDX_C};

%% Scalar coupling: 1J(19F-13C) in Hz
inter.coupling.scalar        = cell(2, 2);
inter.coupling.scalar{1, 2}  =  1e9;%243.5;

%% Redfield relaxation
inter.relaxation  = {'redfield'};
inter.rlx_keep    = 'labframe';
inter.equilibrium = 'zero';
inter.tau_c       = {tau_c};

%% Basis set
bas.formalism     = 'sphten-liouv';
bas.approximation = approx;
%if ~strcmp(approx, 'none')
%    bas.level = 4;
%end

%% Spinach housekeeping
spin_system = create(sys, inter);
spin_system = basis(spin_system, bas);

[H, Q] = hamiltonian(assume(spin_system, 'labframe'));  %#ok<ASGLU>
R = relaxation(spin_system);
K = sparse([], [], [], size(H, 1), size(H, 2));  %#ok<NASGU>

fprintf('Liouville-space dimension: %d\n', size(H, 1));

%% Eigenvalue analysis of relaxation superoperator R
% For the Redfield map to preserve positivity, all eigenvalues of R must
% have non-positive real parts.  Any positive real part signals breakdown.
eigs_R      = eig(full(R));
re_eigs_R   = sort(real(eigs_R), 'descend');
n_positive  = sum(re_eigs_R > 1e-10);

fprintf('R eigenvalues (real parts) — max: %.4f   min: %.4f\n', ...
    re_eigs_R(1), re_eigs_R(end));
if n_positive > 0
    fprintf('  *** %d eigenvalue(s) with positive real part: positivity ALREADY violated ***\n', ...
        n_positive);
    fprintf('  Positive real parts: ');
    fprintf('%.4f  ', re_eigs_R(1:n_positive));
    fprintf('\n');
else
    fprintf('  All eigenvalues non-positive — no immediate breakdown from R alone\n');
end

% -------------------------------------------------------------------------
% NMR single-pulse experiment (preserved for later use)
% -------------------------------------------------------------------------
% parameters.offset      = 0;
% parameters.sweep       = 2000;
% parameters.npoints     = 2048;
% parameters.zerofill    = 4096;
% parameters.pulse_op    = operator(spin_system, 'Lx', '13C');
% parameters.pulse_angle = pi/2;
% parameters.rho0        = state(spin_system, 'Lz', 'all');
% parameters.coil        = state(spin_system, 'L+', '13C');
% parameters.decouple    = {};
% parameters.spc_dim     = 1;
%
% fid = liquid(spin_system, @hp_acquire, parameters, 'labframe');
%
% dt      = 1 / parameters.sweep;
% t_axis  = (0 : parameters.npoints - 1).' * dt;
% lb      = 2;
% fid_apo = fid .* exp(-pi * lb * t_axis);
% phi0    = angle(fid_apo(1));
% phaser  = exp(-1i * phi0);
% spectrum  = fftshift(fft(fid_apo .* phaser, parameters.zerofill));
% freq_axis = linspace(parameters.sweep/2, -parameters.sweep/2, parameters.zerofill);
%
% figure('Position', [100 100 860 420]);
% plot(freq_axis, real(spectrum), 'LineWidth', 1.5);
% xlabel('Frequency (Hz)', 'FontSize', 14);
% ylabel('Intensity',       'FontSize', 14);
% title(sprintf('^{19}F-^{13}C pair  |  \tau_c = %.1e s  |  B_0 = %.1f T  |  approx: %s', ...
%     tau_c, B0, approx), 'FontSize', 13, 'Interpreter', 'tex');
% xlim([-parameters.sweep/2, parameters.sweep/2]);
% grid on; box on;
% set(gca, 'FontSize', 12);
% -------------------------------------------------------------------------

%% Initial state in Spinach Liouville basis
%
% Each |up/down><up/down| = E/2 +/- Lz, so the two-spin product is:
%
%   |ud><ud| = (E/2 + Lz1)(E/2 - Lz2)
%            = (1/4)E@E + (1/2)Lz1@E - (1/2)E@Lz2 - Lz1@Lz2
%
%   |uu><uu| = (E/2 + Lz1)(E/2 + Lz2)
%            = (1/4)E@E + (1/2)Lz1@E + (1/2)E@Lz2 + Lz1@Lz2
switch lower(init_state)
    case 'ud'
        rho0 = (1/4) * state(spin_system, 'E',            'all') ...
             + (1/2) * state(spin_system, 'Lz',           1)     ...
             - (1/2) * state(spin_system, 'Lz',           2)     ...
             -         state(spin_system, {'Lz','Lz'}, {1,2});
    case 'uu'
        rho0 = (1/4) * state(spin_system, 'E',            'all') ...
             + (1/2) * state(spin_system, 'Lz',           1)     ...
             + (1/2) * state(spin_system, 'Lz',           2)     ...
             +         state(spin_system, {'Lz','Lz'}, {1,2});
    otherwise
        error('fc_pair_redfield: init_state must be ''ud'' or ''uu''.');
end

%% Hilbert-space Hamiltonian (4x4) for eigenbasis — isotropic Larmor freqs + J
%
% operator() in sphten-liouv returns 16x16 Liouville superoperators, so we
% build the 4x4 Hilbert-space operators directly from kron products of the
% 2x2 spin-1/2 matrices.  These are kept as separate _h variables to avoid
% confusion with the Liouville-space operator() matrices used for rho0.
gamma_F = spin('19F');
gamma_C = spin('13C');
omega_F = -gamma_F * B0;
omega_C = -gamma_C * B0;

iz = [0.5, 0; 0, -0.5];  ip = [0, 1; 0, 0];  im = [0, 0; 1, 0];  E2 = eye(2);

Lz1_h = kron(iz, E2);  Lz2_h = kron(E2, iz);
Lp1_h = kron(ip, E2);  Lm1_h = kron(im, E2);
Lp2_h = kron(E2, ip);  Lm2_h = kron(E2, im);

J_rad  = 2*pi * inter.coupling.scalar{1, 2};
H_hilb = omega_F*Lz1_h + omega_C*Lz2_h ...
       + J_rad*(Lz1_h*Lz2_h + 0.5*(Lp1_h*Lm2_h + Lm1_h*Lp2_h));

[V, D] = eig(H_hilb);
fprintf('H eigenenergies / (2pi) (Hz): ');
fprintf('%.2f  ', diag(D).' / (2*pi));
fprintf('\n');

%% Classify eigenstates by total spin S and projection M
%
% S^2 = S1^2 + S2^2 + 2*(I1.I2) = 3/2*I_4 + 2*(Lz1*Lz2 + (L+1*L-2 + L-1*L+2)/2)
% Sz  = Lz1 + Lz2
% Solve S(S+1) = <S^2> for S, and M = round(<Sz>).
Sz_mat = Lz1_h + Lz2_h;
S2_mat = 1.5*eye(4) + 2*(Lz1_h*Lz2_h + 0.5*(Lp1_h*Lm2_h + Lm1_h*Lp2_h));

state_labels = cell(1, 4);
for n = 1:4
    S2_exp = real(V(:,n)' * S2_mat * V(:,n));
    Sz_exp = real(V(:,n)' * Sz_mat * V(:,n));
    S_val  = round(0.5 * (-1 + sqrt(max(0, 1 + 4*S2_exp))));
    M_val  = round(Sz_exp);
    if S_val == 0
        state_labels{n} = 'S=0, M=0';
    elseif M_val > 0
        state_labels{n} = sprintf('S=1, M=+%d', M_val);
    elseif M_val == 0
        state_labels{n} = 'S=1, M=0';
    else
        state_labels{n} = sprintf('S=1, M=%d', M_val);
    end
end
fprintf('Eigenstate labels (ascending energy): ');
fprintf(' [%s]', state_labels{:});
fprintf('\n');

%% Liouville <-> Hilbert conversion matrix T  (16x16)
%
% For each of the 16 product operators, state() returns a sparse vector
% whose largest entry sits at the Spinach basis index for that operator.
% Setting T(:, idx) = vec(B_hilbert) / sv(idx) ensures that
%   rho_H = reshape(T * rho_liouv, 4, 4)
% is correct for any Spinach normalization of the basis vectors.
% The third column uses the _h (4x4) Hilbert-space matrices.

n_hilb  = 4;
n_liouv = size(H, 1);

prod_ops = {
    'E',         'all',   eye(4);
    'Lz',        1,       Lz1_h;
    'Lz',        2,       Lz2_h;
    'L+',        1,       Lp1_h;
    'L-',        1,       Lm1_h;
    'L+',        2,       Lp2_h;
    'L-',        2,       Lm2_h;
    {'Lz','Lz'}, {1,2},   Lz1_h*Lz2_h;
    {'L+','Lz'}, {1,2},   Lp1_h*Lz2_h;
    {'L-','Lz'}, {1,2},   Lm1_h*Lz2_h;
    {'Lz','L+'}, {1,2},   Lz1_h*Lp2_h;
    {'Lz','L-'}, {1,2},   Lz1_h*Lm2_h;
    {'L+','L+'}, {1,2},   Lp1_h*Lp2_h;
    {'L+','L-'}, {1,2},   Lp1_h*Lm2_h;
    {'L-','L+'}, {1,2},   Lm1_h*Lp2_h;
    {'L-','L-'}, {1,2},   Lm1_h*Lm2_h;
};

T_liouv2hilb = zeros(n_hilb^2, n_liouv);
for row = 1:size(prod_ops, 1)
    sv = full(state(spin_system, prod_ops{row,1}, prod_ops{row,2}));
    [sv_max, idx] = max(abs(sv));
    if sv_max > 1e-12
        T_liouv2hilb(:, idx) = prod_ops{row,3}(:) / sv(idx);
    end
end

liouv2hilb = @(v) reshape(T_liouv2hilb * full(v), n_hilb, n_hilb);

%% Liouville-space co-vectors for each eigenstate projector |n><n|
%
% P_n(t) = Tr(|n><n| * rho_H(t))
%        = vec(|n><n|)' * vec(rho_H(t))
%        = vec(|n><n|)' * T * rho_liouv(t)
%        = (T' * vec(|n><n|))' * rho_liouv(t)
%
% So proj_liouv(:,n) = T' * vec(|n><n|).
% Using T' (transpose) here, NOT T^{-1}: these differ whenever T is not
% unitary, which is the case for Spinach's unnormalized product operator
% basis (e.g. ||E_4||_F = 2, not 1).
proj_liouv = zeros(n_liouv, n_hilb);
for n = 1:n_hilb
    rho_n = V(:,n) * V(:,n)';
    proj_liouv(:, n) = T_liouv2hilb' * rho_n(:);
end

%% Time grid: span 10 correlation times
t_end   = 10 * tau_c;
n_steps = 500;
t_axis  = linspace(0, t_end, n_steps);
dt      = t_axis(2) - t_axis(1);

%% Propagation under Redfield generator G = -i*H + R
G     = -1i*H + R;
P     = expm(full(G) * dt);    % one-step propagator (reused each step)
rho_t = full(rho0);

populations = zeros(n_hilb, n_steps);
min_eig_rho = zeros(1, n_steps);

for k = 1:n_steps
    for n = 1:n_hilb
        populations(n, k) = real(proj_liouv(:,n)' * rho_t);
    end
    rho_H = liouv2hilb(rho_t);
    min_eig_rho(k) = min(real(eig(rho_H)));
    rho_t = P * rho_t;
end

%% Plot

% Adaptive time axis unit based on t_end
if t_end < 1e-6
    t_scale = 1e9;  t_unit = 'ns';
elseif t_end < 1e-3
    t_scale = 1e6;  t_unit = '\mus';
else
    t_scale = 1e3;  t_unit = 'ms';
end

% Style lookup: {label, RGB color, linestyle, marker, linewidth}
% Singlet uses solid black; triplet states use color + distinct line/marker.
style_lut = {
    'S=0, M=0',  [0.00, 0.00, 0.00], '-',   'none', 2.0;
    'S=1, M=+1', [0.00, 0.45, 0.74], '--',  'none', 1.8;
    'S=1, M=0',  [0.85, 0.33, 0.10], ':',   's',    2.0;
    'S=1, M=-1', [0.47, 0.67, 0.19], '-.',  '^',    1.8;
};
mk_idx = round(linspace(1, n_steps, 20));  % sparse marker positions

fig = figure('Position', [100 100 960 520]);

subplot(2,1,1);
hold on;
for n = 1:n_hilb
    row = find(strcmp(style_lut(:,1), state_labels{n}));
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
legend('FontSize', 11, 'Location', 'best');
title(sprintf(['^{19}F-^{13}C | \\rho_0 = |%s\\rangle | ' ...
    '\\tau_c = %.1e s | B_0 = %.2f T | approx: %s'], ...
    init_state, tau_c, B0, approx), ...
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
    sprintf('fc_pair_%s_tc%.0e_B%.2f_%s.pdf', init_state, tau_c, B0, approx));
if exist('exportgraphics', 'file')
    exportgraphics(fig, pdf_name, 'ContentType', 'vector');
else
    print(fig, pdf_name, '-dpdf', '-bestfit');
end
fprintf('Figure saved: %s\n', pdf_name);

end
