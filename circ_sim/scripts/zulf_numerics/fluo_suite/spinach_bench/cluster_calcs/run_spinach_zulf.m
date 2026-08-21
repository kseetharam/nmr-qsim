function run_spinach_zulf(mol_label, bas_level)
% run_spinach_zulf('mol5a_3.0A', 5)
%
% Generic ZULF single-pulse Bloch-Redfield simulation for one fluo_suite
% cross-verification system, at one IK-0 basis-truncation level.
%
% Loads inputs/<mol_label>_spinach_input.mat (produced by
% export_spinach_inputs.py), builds the Spinach spin system with the SAME
% tau_c and field as the Python Lindblad reference
% (circ_sim/scripts/linblad_dyn/fluo_suite_operators.py), runs the
% sudden-transfer + hard-pulse ZULF protocol under Bloch-Redfield
% relaxation theory, and saves the FID + spectrum + metadata to ../data/.
%
% This is a DELIBERATELY DIFFERENT simulation method from the rest of
% this codebase's Lindblad jump operators (Spinach's own Redfield theory,
% built from the same tau_c/coordinates/shielding/J, rather than the
% extreme-narrowing jump operators in fluo_suite_operators.py) -- the
% point is cross-verification, not re-checking the same derivation.
%
% Physical/acquisition conventions match the two Spinach ZULF scripts
% already run in this repo:
%   circ_sim/data/big_fluo_mols/gemcitabine_5spin_ZULF_13C.m
%   circ_sim/data/big_fluo_mols/fluticasone_ZULF_13C_single.m
%
% Usage (also see submit_spinach_bench_array.sh):
%   matlab -batch "run_spinach_zulf('mol5a_3.0A', 5)"

if ischar(bas_level) || isstring(bas_level)
    bas_level = str2double(bas_level);
end

HERE = fileparts(mfilename('fullpath'));
in_path = fullfile(HERE, 'inputs', [char(mol_label) '_spinach_input.mat']);
data = load(in_path);

n = numel(data.isotopes);
isotopes = cellstr(data.isotopes);

% --- Spin system ---
sys.isotopes = isotopes;
sys.magnet   = data.B_vec_T(3);   % on-axis field, T -- matches Python B_vec_T
sys.output   = 'hush';
sys.disable  = {'hygiene'};

inter.coordinates   = cell(1, n);
inter.zeeman.matrix = cell(1, n);
for i = 1:n
    inter.coordinates{i}   = data.coords_ang(i, :);
    inter.zeeman.matrix{i} = squeeze(data.sigma_ppm_3d(i, :, :));
end

% Only one triangular half is populated (matches data.J_hz_upper, which
% is strictly upper-triangular); create() symmetrizes automatically.
inter.coupling.scalar = cell(n, n);
for i = 1:n
    for j = (i+1):n
        if data.J_hz_upper(i, j) ~= 0
            inter.coupling.scalar{i, j} = data.J_hz_upper(i, j);
        end
    end
end

% --- Basis set: IK-0 restricted Liouville space, level under test ---
bas.formalism     = 'sphten-liouv';
bas.approximation = 'IK-0';
bas.level         = bas_level;

% --- Relaxation theory ---
% tau_c/field fixed across the whole fluo_suite generation pass (see
% circ_sim/scripts/linblad_dyn/fluo_suite_operators.py); temperature
% matches the two prior Spinach ZULF scripts in this repo.
inter.relaxation  = {'redfield'};
inter.equilibrium = 'zero';
inter.rlx_keep    = 'labframe';
inter.tau_c       = {data.tau_c_s};
inter.temperature = 298;

spin_system = create(sys, inter);
spin_system = basis(spin_system, bas);

% --- Gamma-weighted operators (sudden-transfer state, pulse, coil) ---
weights = spin_system.inter.gammas / spin('1H');

rho_sud    = sparse(0);
Sy         = sparse(0);
coilXYquad = sparse(0);
for k = 1:spin_system.comp.nspins
    rho_sud    = rho_sud    + weights(k) * state(spin_system, {'Lz'}, {k});
    Sy         = Sy         + weights(k) * (operator(spin_system, {'L+'}, {k}) - ...
                                             operator(spin_system, {'L-'}, {k})) / 2i;
    coilXYquad = coilXYquad + weights(k) * state(spin_system, {'L+'}, {k});
end

R = relaxation(spin_system);  %#ok<NASGU>  -- primes the relaxation superoperator (matches prior scripts)

% --- Single-pulse ZULF experiment ---
% Acquisition settings match gemcitabine_5spin_ZULF_13C.m /
% fluticasone_ZULF_13C_single.m for direct comparability.
parameters.offset      = 0;
parameters.sweep       = 3000;
parameters.npoints     = 4*1024;
parameters.zerofill    = 4*1024;
parameters.rho0        = step(spin_system, Sy, rho_sud, pi/2);   % 90-deg Y pulse
parameters.coil        = coilXYquad;                              % quadrature detection
parameters.spins       = {'1H'};
parameters.invert_axis = 0;
parameters.axis_units  = 'Hz';

fid_raw  = liquid(spin_system, @acquire, parameters, 'labframe');
fid_apod = apodisation(spin_system, fid_raw, {{'exp', 3}});
spec     = fftshift(fft(fid_apod, parameters.zerofill));
freq     = (-parameters.zerofill/2 : parameters.zerofill/2-1) * (parameters.sweep/parameters.zerofill);

% --- Save ---
out_dir = fullfile(HERE, '..', 'data');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

fid_raw_real = real(fid_raw); fid_raw_imag = imag(fid_raw);
spec_real    = real(spec);    spec_imag    = imag(spec);

metadata = struct( ...
    'mol_id',        char(data.mol_id), ...
    'drug_name',     char(data.drug_name), ...
    'anchor_carbon', data.anchor_carbon, ...
    'cutoff_ang',    data.cutoff_ang, ...
    'n_spins',       n, ...
    'isotopes',      {isotopes}, ...
    'atom_indices_source_workbook', data.atom_indices_source_workbook, ...
    'bas_formalism',     bas.formalism, ...
    'bas_approximation', bas.approximation, ...
    'bas_level',         bas_level, ...
    'relaxation_theory', 'redfield', ...
    'tau_c_s',           data.tau_c_s, ...
    'B_vec_T',           data.B_vec_T, ...
    'temperature_K',     298, ...
    'sweep_hz',   parameters.sweep, ...
    'npoints',    parameters.npoints, ...
    'zerofill',   parameters.zerofill, ...
    'apodisation', 'exp,3', ...
    'nmr_protocol', ['Sudden-transfer + hard-pulse ZULF. rho_sud = sum_i w_i Lz_i; ' ...
                      'rho0 = exp(-i pi/2 Sy) rho_sud exp(+i pi/2 Sy); ' ...
                      'coil = sum_i w_i L+_i; FID(t) = Tr(coil . rho(t)).'], ...
    'input_file',   in_path, ...
    'generated_by', 'circ_sim/scripts/zulf_numerics/fluo_suite/spinach_bench/cluster_calcs/run_spinach_zulf.m');

out_name = sprintf('%s_bas%d_fid.mat', char(mol_label), bas_level);
out_path = fullfile(out_dir, out_name);
save(out_path, 'fid_raw_real', 'fid_raw_imag', 'spec_real', 'spec_imag', 'freq', 'metadata', '-v7.3');
fprintf('Saved -> %s\n', out_path);

end
