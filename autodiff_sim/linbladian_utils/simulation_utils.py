import os
from timeit import default_timer as timer
import pickle
import numpy as np
from scipy.linalg import expm
import scipy
import sys

from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit import MachineModel
from bqskit.ir.gates import CZGate, RZGate, SXGate
from bqskit import compile
from cirq.contrib.qasm_import import circuit_from_qasm
import cirq

sys.path.append('./')
from basis_utils import S_plus, S_minus, MatRepLib, InnProd,Sz,Sx,Sy
import openfermion as of
from openfermion import QubitOperator

import logging
from bqskit.compiler import Compiler
#from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
#from bqskit.passes import QFASTDecompositionPass
#from bqskit.passes import ScanningGateRemovalPass
#from bqskit.passes import ToU3Pass
#from bqskit.passes import ToVariablePass
#from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QSearchSynthesisPass
#from bqskit.passes import UnfoldPass
from bqskit.passes import SimpleLayerGenerator
from bqskit.ir.gates import VariableUnitaryGate
from bqskit import Circuit


def loadMat(mol,path):
    fname=path+f'generators_noesy_{mol}.mat'

    return spio.loadmat(fname, squeeze_me=True)

def EmbedInU(TarMat):

    Dim = TarMat.shape[0]

    UR = scipy.linalg.sqrtm(np.eye(Dim)-np.dot(TarMat,TarMat.conjugate().T))
    LL = scipy.linalg.sqrtm(np.eye(Dim)-np.dot(TarMat.conjugate().T,TarMat))
    
    U_meth = np.zeros([2*Dim,2*Dim],dtype=complex)
    U_meth[0:Dim,0:Dim] = TarMat
    U_meth[0:Dim,Dim:2*Dim]=UR
    U_meth[Dim:2*Dim,0:Dim]=LL
    U_meth[Dim:2*Dim,Dim:2*Dim]=-TarMat.conjugate().T

    return U_meth

def Umetric(TarMat):
    dim = TarMat.shape[0]
    
    return np.linalg.norm(np.dot(TarMat.conj().T,TarMat)-np.eye(dim))


def run_simp_layer_flow_example(in_circuit,
        amount_of_workers: int = 10, synt_pass = QSearchSynthesisPass
) -> tuple[Circuit, float]:
    
    num_multistarts = 32
   
    instantiate_options = {
        'method': 'qfactor',
        'multistarts': num_multistarts,
    }

    passes = [

        # Split the circuit into partitions
       #QSearchSynthesisPass(instantiate_options=instantiate_options),
       synt_pass(layer_generator=SimpleLayerGenerator(two_qudit_gate=VariableUnitaryGate(2),single_qudit_gate_1=VariableUnitaryGate(1)),
                 success_threshold=1e-3,max_layer=5000,instantiate_options=instantiate_options)
       
       #QSearchSynthesisPass(layer_generator=LayerGenDef.AltLayer(),instantiate_options=instantiate_options)

    ]
    

    with Compiler(
        num_workers=amount_of_workers,
        runtime_log_level=logging.INFO,
    ) as compiler:

        print('Starting flow using QFactor instantiation')
        start = timer()
        out_circuit = compiler.compile(in_circuit, passes)
        end = timer()
        run_time = end - start

    return out_circuit, run_time




def SimulateBlock(bqskit_circ,n_flag,reps=1000,gate_set={CZGate(), RZGate(), SXGate()},noise=None):
    """
    Simulate post-selected samples from a Block-encoding unitary using cirq. Assuming that the target n-qubit matrix is block encoded in the 
    upper-left block of the unitary, this corresponds to post-select the measurement outcomes in the last n-qubits of the circuit.
    Args:
    bqskit_circ: result of circuit synthesis
    n_flag: number of flag qubits for the block encoding
    gate_set: the target gate set to perform the compilation
    """

    model = MachineModel(bqskit_circ.num_qudits, gate_set=gate_set)

    inst_circuit = compile(bqskit_circ, model=model)
    lang = OPENQASM2Language()
    qasm = lang.encode(inst_circuit)

    cirq_circ = circuit_from_qasm(qasm)
    qubits = sorted(cirq_circ.all_qubits())
    
    Nqubs = len(cirq_circ.all_qubits())

    n_sys = Nqubs - n_flag
    #qubits = cirq.LineQubit.range(Nqubs)
    control_register = qubits[0:n_flag]
    target_register = qubits[n_flag:]
    
    
    cirq_circ.append(cirq.measure(*control_register, key='control')) 
    cirq_circ.append(cirq.measure(*target_register, key='target'))
    
    
    simulator = cirq.Simulator()
    result = simulator.run(cirq_circ, repetitions=reps)
    
    # Step 4: Post-select results based on control register measurements
    control_measurements = result.measurements['control']
    target_measurements = result.measurements['target']
    
    # Post-select where control register is [0, 0] (or any desired condition)
    #post_selected_indices = np.where((control_measurements[:, 0] == 0) & (control_measurements[:, 1] == 0))[0]
    post_selected_indices = np.where((control_measurements[:] == [0]*n_flag))[0]
    post_selected_target_measurements = target_measurements[post_selected_indices]

    return post_selected_target_measurements

def EstimatePolarization(Measurements):
    """
    Estimate the expectation value of S_{z} = \sum_{n}\sigma^{(z)}_{n} from a list of Measurements
    """
    nqubs = len(Measurements[0])

    Tot_pol=0.0
    for i in range(len(Measurements)):

        m = 0.0
        for j in range(nqubs):
            res=Measurements[i][j]
            m+=(-1.0)**res

        Tot_pol+=m

    return Tot_pol/len(Measurements)


#####################Set of functions that aid the simulation of NOESY spectra##############################
def int_to_binary(n, bit_length=None):
    """
    Convert an integer to a binary vector, starting with the least significant bit (LSB).
    
    Parameters:
    - n: The integer to be converted.
    - bit_length: The length of the resulting binary vector (optional). 
                  If not provided, the vector will have the minimal number of bits required.
    
    Returns:
    - A list representing the binary vector with the least significant bit first.
    """
    if n < 0:
        raise ValueError("Negative integers are not supported.")
    
    # Get the binary representation of the number (without the '0b' prefix)
    binary_string = bin(n)[2:]
    
    # If bit_length is provided, pad with leading zeros
    if bit_length is not None:
        binary_string = binary_string.zfill(bit_length)
    
    # Convert the string to a list of integers (binary vector) and reverse it for LSB-first order
    return [int(bit) for bit in reversed(binary_string)]


def HamMatRep(H,basis,n_qubits=2):
    N = len(basis)

    Matrep= np.zeros([N,N],dtype=complex)
    for i in range(N):
        for j in range(N):
            Matrep[i,j] = InnProd(basis[i],of.commutator(H,basis[j]),n_qubits=n_qubits)

    
    return Matrep

def GenH0_Ham(offset,B0,zeeman_scalars,Jcoups,gamma):
    """
    Returns: the zeroth order Hamiltonian considered for dynamical evolution in nthe simulations, in OpenFermion format
    Args:
    offset: frequency offset for the spin Zeeman frequencies, in Hz
    B0: Strength of the magnetic field in Teslas
    zeeman_scalars: list of chemical shifts for spins, in ppm
    Jcoups: matrix of size N x N, N being the number of spins, that encodes the scalar couplings between spins (in Hz)
    gamma: gyromagnetic ratio of the spins (an homonuclear case is assumed) 
    """


    #offset = -46681
    #B0 = 9.3933
    w0 = -gamma*B0
    o1 = 2*np.pi*offset
    Nspins = len(zeeman_scalars)

    Hamiltonian = of.QubitOperator()

    for i in range(Nspins):
        w = o1+w0*zeeman_scalars[i]/1e6
        Hamiltonian+=w*Sz(i)
        for j in range(i+1,Nspins):

            Hamiltonian+=2*np.pi*Jcoups[i,j]*(Sx(i)*Sx(j)+Sy(i)*Sy(j)+Sz(i)*Sz(j))
    
    return Hamiltonian

def Embed_JumpOps(ListJumpOps,qub_offset):
    """
    Returns: the Jump operators contained in ListJumpOps as a dilated-Hamiltonian. the ith Jump operator will be embedded as |i><0|Op+h.c., where |i> corresponds to a binary vector
    Args:
    ListJumpOps: the jump operators to be encoded in the dilated Hamiltonian. Note that they should be scaled by sqrt(\kappa), \kappa being the damping rate for this Jump operator
    qub_offset: starting qubit index for ancillary qubits
    """
    NOps = len(ListJumpOps)
    n_ancs = int(np.ceil(np.log2(30)))  #the number of ancillary qubits

    Embed_Ham = QubitOperator()

    for idx in range(len(ListJumpOps)):

        bin_vec = int_to_binary(idx, bit_length=n_ancs)

        Projector = 1*QubitOperator()
        for i in range(n_ancs):
            if bin_vec[i]==1:
                Projector=Projector*S_plus(qub_offset+i)
            else:
                Projector=Projector*S_minus(qub_offset+i)*S_plus(qub_offset+i)

        Embed_Ham += Projector*ListJumpOps[idx] 

    return Embed_Ham


def Gen_Prop_PauliBasis(Ham_qub,JumpOps,basis):
    """
    Returns: 1) a matrix representation of the coherent part of the Linbladian super-operator and 2) the dissipative counterpart. This function is useful to tune the relaxation channels to incorporate 
    during the incoherent evolution and generate reference values to benchmark circuit simulations
    Args:
    Ham_qub, the zeroth order Hamiltonian that describes the evolution, in Openfermion format
    JumpOps: list that contains the jump operators, it is assumed that they are properly scaled to incorporate the damping rates that enter in the Linbladian equations of motion
    basis: normalized list of orthonormal operators used to build the matrix representation of the super-operators
    """

    H = HamMatRep(Ham_qub,basis)

    nqubs = of.count_qubits(Ham_qub)

    N=len(basis)
    R=np.zeros([N,N],dtype=complex)

    for i in range(len(JumpOps)):
        R+=MatRepLib(basis,JumpOps[i],JumpOps[i],n_qubits=nqubs)

    return H,R

#####generation of NOESY spectra...
def sqcosbell_2d_apod(fid_2d):
    fid_2d[0,:] = fid_2d[0,:]/2
    fid_2d[:,0] = fid_2d[:,0]/2
    x = np.linspace(0,np.pi/2,fid_2d.shape[0])
    y = np.linspace(0,np.pi/2,fid_2d.shape[1])

    decay_col = np.square(np.cos(x))
    decay_row = np.square(np.cos(y))

    return fid_2d*np.outer(decay_col,decay_row)

def GenFIDsignals(Ham,R,Tpts1,Tpts2,rho0,coil,tmix,dt1,dt2,Lx,Ly):
    #Dim = Ham.shape[0]
    Lnet = Ham+1j*R 
    L_dt1 = expm(-1j*Lnet*dt1)
    L_dt2 = expm(-1j*Lnet*dt2)
    pulse_mix = expm(-1j*Lnet*tmix)


    pulse_90x = expm(-1j*Lx*np.pi/2)
    pulse_90y = expm(-1j*Ly*np.pi/2)
    pulse_90mx = expm(1j*Lx*np.pi/2)
    pulse_90my = expm(1j*Ly*np.pi/2)


    #FID_1 = np.zeros([Tpts2,Tpts1],dtype=complex)
    #FID_2 = np.zeros([Tpts2,Tpts1],dtype=complex)
    #FID_3 = np.zeros([Tpts2,Tpts1],dtype=complex)
    #FID_4 = np.zeros([Tpts2,Tpts1],dtype=complex)

    #First 90x pulse:
    rho_t = np.copy(rho0)
    rho_t = np.dot(pulse_90x,rho_t)

    rho_stack = []
    rho_stack.append(rho_t)

    rho_temp = np.copy(rho_t)
    for i in range(1,Tpts1):
        rho_temp = np.dot(L_dt1,rho_temp)
        rho_stack.append(rho_temp)


    rho_stack1_1 = []
    rho_stack1_2 = []
    rho_stack1_3 = []
    rho_stack1_4 = []

    for i in range(Tpts1):
        rho_stack1_1.append(pulse_90y@pulse_mix@pulse_90x@rho_stack[i])
        rho_stack1_2.append(pulse_90y@pulse_mix@pulse_90y@rho_stack[i])
        rho_stack1_3.append(pulse_90y@pulse_mix@pulse_90mx@rho_stack[i])
        rho_stack1_4.append(pulse_90y@pulse_mix@pulse_90my@rho_stack[i])


    fid_temp_1 = np.zeros([Tpts2,Tpts1],dtype=complex)
    fid_temp_2 = np.zeros([Tpts2,Tpts1],dtype=complex)
    fid_temp_3 = np.zeros([Tpts2,Tpts1],dtype=complex)
    fid_temp_4 = np.zeros([Tpts2,Tpts1],dtype=complex)

    for i in range(Tpts1):
        rho1 = rho_stack1_1[i]
        rho2 = rho_stack1_2[i]
        rho3 = rho_stack1_3[i]
        rho4 = rho_stack1_4[i]

        for j in range(Tpts2):
            fid_temp_1[j,i] = np.dot(coil,rho1)
            rho1 = L_dt2@rho1

            fid_temp_2[j,i] = np.dot(coil,rho2)
            rho2 = L_dt2@rho2

            fid_temp_3[j,i] = np.dot(coil,rho3)
            rho3 = L_dt2@rho3

            fid_temp_4[j,i] = np.dot(coil,rho4)
            rho4 = L_dt2@rho4
    
    return fid_temp_1, fid_temp_2, fid_temp_3, fid_temp_4




def GenNOESYSpectrum(Ham,R,Tpts1,Tpts2,rho0,coil,tmix,dt1,dt2,zerofill1,zerofill2,Lx,Ly,returnFID=True):

    fid_temp_1, fid_temp_2, fid_temp_3, fid_temp_4 = GenFIDsignals(Ham,R,Tpts1,Tpts2,rho0,coil,tmix,dt1,dt2,Lx,Ly)
    
    fid_test_cos = fid_temp_1 - fid_temp_3
    fid_test_sin = fid_temp_2 - fid_temp_4

    fid_cos = sqcosbell_2d_apod(fid_test_cos)
    fid_sin = sqcosbell_2d_apod(fid_test_sin)

    f1_cos = np.real(np.fft.fftshift(np.fft.fft2(fid_cos,[zerofill2],[0]),[0]))
    f1_sin = np.real(np.fft.fftshift(np.fft.fft2(fid_sin,[zerofill2],[0]),[0]))


    f1_states = f1_cos-1j*f1_sin

    spectrum = np.fft.fftshift(np.fft.fft2(f1_states,[zerofill1],[1]),[1])
    if returnFID:
        ###NOTE: return the FID witouth post-processing
        return spectrum, fid_test_cos-1j*fid_test_sin
    else:
        return spectrum

###Analysis and comparison of spectra...
def NormalizeSpectrum(Spectrum,Tpts1,Tpts2,dt1,dt2):
    Pos_spec= np.abs(np.real(Spectrum))

    Int_spec = 0.0 

    deltw1 = 2*np.pi/(Tpts1*dt1)
    deltw2 = 2*np.pi/(Tpts2*dt2)

    for i in range(Pos_spec.shape[0]):
        for j in range(Pos_spec.shape[1]):
            Int_spec+=Pos_spec[i,j]

    Int_spec = Int_spec*deltw1*deltw2

    Norm_spec = Pos_spec/Int_spec
    return Norm_spec

def Hellinger_2D(Spec1,Spec2,Tpts1,Tpts2,dt1,dt2):
    """
    Compute the Hellinger distance between Spec1 and Spec2 spectra 
    """

    ####Normalize the spectra...

    NSpec1 = NormalizeSpectrum(Spec1,Tpts1,Tpts2,dt1,dt2)
    NSpec2 = NormalizeSpectrum(Spec2,Tpts1,Tpts2,dt1,dt2)


    Nx = Spec1.shape[0]
    Ny = Spec1.shape[1]
    
    deltw1 = 2*np.pi/(Tpts1*dt1)
    deltw2 = 2*np.pi/(Tpts2*dt2)

    diff = 0.0
    for i in range(Nx):
        for j in range(Ny):
            diff+=(np.sqrt(NSpec1[i,j])-np.sqrt(NSpec2[i,j]))**2
    
    return diff*deltw1*deltw2


