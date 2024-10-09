% Phase-sensitive homonuclear NOESY pulse sequence. Syntax:
%
%             fid=noesy(spin_system,parameters,H,R,K)
%
% Parameters:
%
%    parameters.sweep     sweep widths, Hz
%
%    parameters.npoints   number of points for both dimensions
%
%    parameters.spins     nuclei on which the sequence runs,
%                         specified as '1H', '13C', etc.
%
%    parameters.tmix      mixing time, seconds
%
%    parameters.decouple  spins to be decoupled, specified either
%                         by name, e.g. {'13C','1H'}, or by a list
%                         of numbers, e.g. [1 2]
%
%    parameters.rho0      initial state; skip this and specify
%                         parameters.needs={'rho_eq'} to start
%                         from exact thermal equilibrium
%
%    H - Hamiltonian matrix, received from context function
%
%    R - relaxation superoperator, received from context function
%
%    K - kinetics superoperator, received from context function
%
% Outputs:
%
%    fid.cos,fid.sin - two components of the FID for F1 hyper-
%                      complex processing
%
% Note: this function is used for extreme simulations (proteins
%       and nucleic acids) - its layout is optimised for minimum
%       memory footprint rather than CPU time.
%
% i.kuprov@soton.ac.uk
% luke.edwards@ucl.ac.uk
% hannah.hogben@chem.ox.ac.uk
%
% <https://spindynamics.org/wiki/index.php?title=noesy.m>

function [rhot, obs_p, obs_z]=noesy_trajectory(spin_system,parameters,H,R,K)

% Consistency check
grumble(spin_system,parameters,H,R,K);

% Coherent evolution timestep
timestep=1./parameters.sweep;

% Detection state
coil_p=state(spin_system,'L+',parameters.spins{1},'cheap');
coil_z=state(spin_system,'Lz',parameters.spins{1},'cheap');

% Pulse operators
Lp=operator(spin_system,'L+',parameters.spins{1});
Lx=(Lp+Lp')/2; Ly=(Lp-Lp')/2i;

% Decoupling
if isfield(parameters,'decouple')
    [H,parameters.rho0]=decouple(spin_system,H,parameters.rho0,...
                                               parameters.decouple);
end

% First pulse
rho=step(spin_system,Lx,parameters.rho0,pi/2);

% Phase cycle specification
Op2={Lx,Ly,Lx,Ly}; An2={+pi/2,+pi/2,-pi/2,-pi/2};
Op3={Ly,Ly,Ly,Ly}; An3={+pi/2,+pi/2,+pi/2,+pi/2};

% Trajectory phase cycle
rho_cell=cell(1,4);

% Phase cycle loop
for n=1:4

    % F1 evolution
    rho_stack_F1=evolution(spin_system,H+1i*R+1i*K,[],rho,timestep(1),...
                        parameters.npoints(1)-1,'trajectory');
    % Second pulse
    rho_stack_P1=step(spin_system,Op2{n},rho_stack_F1,An2{n});

    % % Homospoil
    % rho_stack_P1=homospoil(spin_system,rho_stack_P1,'destroy');

    % Mixing time
    rho_stack_M=evolution(spin_system,1i*R+1i*K,[],...
                        rho_stack_P1(:,end),parameters.tmix./parameters.npoints_mix,parameters.npoints_mix-1,'trajectory');

    % % Homospoil
    % rho_stack_P2=homospoil(spin_system,rho_stack_M,'destroy');
      rho_stack_P2 = rho_stack_M;

    % Third pulse
    rho_stack_P2=step(spin_system,Op3{n},rho_stack_P2,An3{n});

    % F2 evolution
    rho_stack_F2=evolution(spin_system,H+1i*R+1i*K,[],rho_stack_P2(:,end),...
                      timestep(2),parameters.npoints(2)-1,'trajectory');
    
    % Concatenate evolution
    rho_cell{n} = [rho_stack_F1, rho_stack_M, rho_stack_F2];
end

% % Axial peak elimination
% fid.cos=fids{1}-fids{3}; fid.sin=fids{2}-fids{4};

rhot = rho_cell{1};
obs_p=coil_p'*rhot;
obs_z=coil_z'*rhot;

end

% Consistency enforcement
function grumble(spin_system,parameters,H,R,K)
if ~ismember(spin_system.bas.formalism,{'sphten-liouv','zeeman-liouv'})
    error('this function is only available for sphten-liouv and zeeman-liouv formalisms.');
end
if (~isnumeric(H))||(~isnumeric(R))||(~isnumeric(K))||...
   (~ismatrix(H))||(~ismatrix(R))||(~ismatrix(K))
    error('H, R and K arguments must be matrices.');
end
if (~all(size(H)==size(R)))||(~all(size(R)==size(K)))
    error('H, R and K matrices must have the same dimension.');
end
if ~isfield(parameters,'sweep')
    error('sweep width should be specified in parameters.sweep variable.');
elseif numel(parameters.sweep)~=2
    error('parameters.sweep array should have exactly two elements.');
end
if ~isfield(parameters,'spins')
    error('working spins should be specified in parameters.spins variable.');
elseif numel(parameters.spins)~=1
    error('parameters.spins cell array should have exactly one element.');
end
if ~isfield(parameters,'npoints')
    error('number of points should be specified in parameters.npoints variable.');
elseif numel(parameters.npoints)~=2
    error('parameters.npoints array should have exactly two elements.');
end
if ~isfield(parameters,'tmix')
    error('mixing time should be specified in parameters.tmix variable.');
elseif numel(parameters.tmix)~=1
    error('parameters.tmix array should have exactly one element.');
end
end

% According to a trade legend, Anil Kumar had to run the very first
% NOESY experiment on a Saturday -- his supervisors viewed it as a
% waste of valuable spectrometer time.

