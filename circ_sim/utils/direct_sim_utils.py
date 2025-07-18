###utils for FID simulation of NOESY spectrum based on sampling initial states and proppagating in time...
import cirq
import openfermion as of
import sys
sys.path.append('./')
from basis_utils import InnProd
from qiskit import transpile, QuantumCircuit
from qiskit.qasm2 import dumps
from cirq.contrib.qasm_import import circuit_from_qasm
import scipy
from openfermion import QubitOperator
from basis_utils import S_plus, S_minus, MatRepLib
from basis_utils import read_spinach_info, build_list_ISTs, NormalizeBasis, MatRepLib,Sz,Sx,Sy

import numpy as np
from scipy.linalg import expm
import copy


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


def SyntCircuitWarp(Unitary,nqubs):

    qc = QuantumCircuit(nqubs)
    qc.unitary(Unitary, range(nqubs))

    # Optimize the circuit using transpile
    optimized_circuit = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=2)

    #return optimized_circuit
    qasm_str = dumps(optimized_circuit)

    return circuit_from_qasm(qasm_str)


def Estimate_Z_fromMeas(measurements):

    m = 0.0
    for i in range(len(measurements)):
        #print(measurements[i][0])
        m += (-1.0)**float(measurements[i][0])

    return m/len(measurements)


def EstMag_from_Circ(Unitary,nqubs,n_anc,samples=1000):
    """
    Simulate circuit (by wrapping the unitary Unitary) that embeds the jump operators through a dilated Hamiltonian and sample the total magnetization over the nqubs qubits
    and tracing out over n_anc ancillary qubits
    """
    custom_gate = cirq.MatrixGate(Unitary)

    # Define three qubits
    qubits = cirq.LineQubit.range(nqubs+n_anc)
    simulator = cirq.Simulator()

    Expect_Zs = []

    #Question: can we get away with estimation expectation value of a single Z_{i} and the total magnetization is
    #N*Z_{i}?

    for i in range(nqubs):
        circuit = cirq.Circuit(custom_gate.on(*qubits))
        system_register = qubits[i]
        anc_register = qubits[nqubs:(nqubs+n_anc)]
    
        circuit.append(cirq.measure(system_register, key='sys')) 
        circuit.append(cirq.measure(*anc_register, key='anc'))

        result = simulator.run(circuit, repetitions=samples)

        sys_qub_meas = result.measurements['sys']
        #anc_meas = result.measurements['anc']
        Expect_Zs.append(Estimate_Z_fromMeas(sys_qub_meas))

        
    return np.sum(Expect_Zs)

def Get_samps_DilHam(Unitary,nqubs,n_anc,idx,samples=1000):
    """
    Simulate the Unitary with a cirq wrapper and obtain samples.
    Args:
    Unitary, the target unitary to simulate
    nqubs:  number of qubits that comprise the spin system
    n_anc: number of ancillary qubits that we need to trace over 
    idx: this is the index of the system qubit we sample over (together with the ancilla qubit, to perform the partial trace)
    """

    custom_gate = cirq.MatrixGate(Unitary)

    # Define three qubits
    qubits = cirq.LineQubit.range(nqubs+n_anc)
    simulator = cirq.Simulator()


    circuit = cirq.Circuit(custom_gate.on(*qubits))
    system_register = qubits[idx]
    anc_register = qubits[nqubs:(nqubs+n_anc)]

    circuit.append(cirq.measure(system_register, key='sys')) 
    circuit.append(cirq.measure(*anc_register, key='anc'))

    result = simulator.run(circuit, repetitions=samples)

    sys_qub_meas = result.measurements['sys']
    
    return sys_qub_meas



#For the 2 spin system we can sample over all the product states that are eigenstates of Sz_{Tot}

def Calc_FID(HamU,pulseU,Tsteps, nqubs=2,n_anc=1,samples=1000,InitStPrepOps=[QubitOperator('X0 X1')]):
    """
    Calculates the FID signal from circuit simulations. Considering the simple 2 level system, we perform the simulations considering the 4 initial states |00>,|01>,|10>,|11> 
    and average the expectation value of the total magnetization.
    """
    
    #Unitary for Hamiltonian simulation of the dilated Hamiltonian...

    U_H = np.copy(HamU) #initial pulse 
    for i in range(1,Tsteps):
        U_H = U_H@HamU
    
    #simulation for initial state |00>
    SimU = U_H@pulseU

    Av_Mag = 0.0
    Av_Mag += 2*EstMag_from_Circ(SimU,nqubs,n_anc,samples=samples) #The factor 2 is the eigenvalue of Sz_tot for eigenstate |00>
    print("Magnetization after sampling vacuum is:",Av_Mag)

    sp_ExcOp = of.get_sparse_operator(InitStPrepOps[0],n_qubits=(nqubs+n_anc))
    SimU = U_H@pulseU@sp_ExcOp.toarray()
    Av_Mag+=-2*EstMag_from_Circ(SimU,nqubs,n_anc,samples=samples)

    #Simulation for states |01>,|10>,|11>
    #for i in range(3):
    #    sp_ExcOp = of.get_sparse_operator(InitStPrepOps[i],n_qubits=(nqubs+n_anc))
    #    SimU = U_H@pulseU@sp_ExcOp.toarray()

    #    Av_Mag+=EstMag_from_Circ(SimU,nqubs,n_anc,samples=samples)

    return Av_Mag

def TrotStepLinb(Uprop,qubits,anc_register,anc_label=0,measure_anc=True):
    """ 
    Construct circuit for a Trotter step, using the unitary propagator corresponding to the dilated Hamiltonian using n_anc qubits.
    Args:
    qubs, the Cirq Linear qubit array considered for the whole simulation
    anc_register, array of qubits that correspond to the ancillary register
    measure_anc: boolean that indicates whether we append measurement operations to the ancilla qubit.
    """
    
    custom_gate = cirq.MatrixGate(Uprop)
    circuit = cirq.Circuit(custom_gate.on(*qubits))

    if measure_anc:
        circuit.append(cirq.measure(*anc_register,key='anc'+str(anc_label)))  # Me
        circuit.append(cirq.ResetChannel()(*anc_register))

    return circuit

def BuildUCirc(Unitary,qubits):
    """
    Construct a unitary that wraps the Unitary, acting on the Cirq linear qubit array qubits 
    """
    custom_gate = cirq.MatrixGate(Unitary)
    circuit = cirq.Circuit(custom_gate.on(*qubits))

    return circuit


def Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit,Uprop1,Uprop2,CircPulse1,CircPulse2,Circpulse3,qubits,anc_register):
    """
    Build the circuit for simulation of the FID, it does not include the measurement on the system qubits at the end, with the purpose of
    reusing this circuit.
    Args:
    t1_steps: number of Trotter steps for the first free time evolution phase of the protocol
    t2_steps: number of Trotter steps for the final free time evolution phase of the protocol
    tmix_steps: number of Trotter steps for the mixing free time evolution in the protocol
    CircInit: the circuit that prepares the initial state to carry out time evolution on
    Uprop1: unitary for the Trotter step in simulation of the first free time evolution phase of the protocol
    Uprop2: unitary for the Trotter step in simulation of the final free time evolution phase of the protoco
    CircPulse1: circuit for simulation of first pulse of the protocol
    CircPulse2: circuit for simulation of second pulse of the protocol 
    Circpulse3: circuit for simulation of third pulse of the protocol 
    qubits: Circ Linear qubit array that contains the system and ancillary qubits
    anc_register: Circ Linear array that corresponds to the ancilla qubit register 
    """
    #build the Trotterized circuits that describe open-system evolution...
    circuit = cirq.Circuit()
    circuit.append(CircInit)

    Nqubs = len(qubits) - len(anc_register)

    circuit.append(CircPulse1)

    count_anc = 0
    for i in range(t1_steps):
        TrotCirc1 = TrotStepLinb(Uprop1,qubits,anc_register,anc_label=count_anc)
        circuit.append(TrotCirc1)
        count_anc+=1
    
    circuit.append(CircPulse2)
    
    for i in range(tmix_steps): #Notice that use the Trotter time step encoded in TrotCirc1 as a "unit" for simulation for tmix
        TrotCirc1 = TrotStepLinb(Uprop1,qubits,anc_register,anc_label=count_anc)
        circuit.append(TrotCirc1)
        count_anc+=1

    circuit.append(Circpulse3)

    for i in range(t2_steps):
        TrotCirc2 = TrotStepLinb(Uprop2,qubits,anc_register,anc_label=count_anc)
        circuit.append(TrotCirc2)
        count_anc+=1

    #circuit.append(cirq.inverse(CircPulse1)) #To measure in the computational basis
    #For the estimation of the FID signal, we need to sample in the X and Y basis, therefore, we append Hadamard and S gates to obtain two different circuits, thta can be
    #sampled in the computational basis...
    circuitX = copy.deepcopy(circuit)
    circuitY = copy.deepcopy(circuit)

    for i in range(Nqubs):
        circuitX.append(cirq.H(qubits[i]))
        circuitY.append(cirq.inverse(cirq.S)(qubits[i]))
        circuitY.append(cirq.H(qubits[i]))

    
    return circuitX, circuitY
    
def Meas_TotMag(Circ,qubits,nqubs,samples):
    """ 
    Simulate the circuit Circ by performing measurements on each of the qubits that define the system qubit register
    
    """

    #custom_gate = cirq.MatrixGate(Unitary)

    # Define three qubits
    #qubits = cirq.LineQubit.range(nqubs+n_anc)
    simulator = cirq.Simulator()

    Expect_Zs = []

    #Question: can we get away with estimation expectation value of a single Z_{i} and the total magnetization is
    #N*Z_{i}?

    for i in range(nqubs):
        
        curr_circ = copy.deepcopy(Circ)

        system_register = qubits[i]
        #anc_register = qubits[nqubs:(nqubs+n_anc)]
    
        curr_circ.append(cirq.measure(system_register, key='sys')) 
        #circuit.append(cirq.measure(*anc_register, key='anc'))

        result = simulator.run(curr_circ, repetitions=samples)

        sys_qub_meas = result.measurements['sys']
        #anc_meas = result.measurements['anc']
        Expect_Zs.append(Estimate_Z_fromMeas(sys_qub_meas))

    return np.sum(Expect_Zs)

def Meas_TotMag2(Circ,qubits,nqubs,samples):
    """ 
    More efficient estimation of total magnetization than that in Meas_TotMag. We measure simultaneously all system qubits, and post-process the results for estimation of total magnetization
    instead of using different versions of the circuit for estimation of each local observable. This can be done due to the shared eigenbasis of the total magnetization and Z0Z1...Z_{N-1}
    """
    simulator = cirq.Simulator()

    for i in range(nqubs):
        Circ.append(cirq.measure(qubits[i],key='sys'+str(i)))
    
    result = simulator.run(Circ, repetitions=samples)

    Expect_Zs = []
    for i in range(nqubs):
        sys_qub_meas = result.measurements['sys'+str(i)]
        Expect_Zs.append(Estimate_Z_fromMeas(sys_qub_meas))
        print("Expect value for spin", i, "is", Expect_Zs[i])
    
    return np.sum(Expect_Zs)


###TODO: now that we have the fucntions that generalize the construction of the dilated Hamiltonian to many ancillary qubits, we can provide a cleaner 
#version of this function....

def CircGen2D_FID(Ham,JumpOps,Tpts1,Tpts2,tmix,dt1,dt2,Lx,Ly,samples=1000):
    """
    Function to sample the 2D FID signals needed for calculation of 2D spectrum in a NOESY experiment.
    Args:
    Ham: the Hamiltonian in OpenFermion format to be simulated
    JumpOps: list of OpenFermion jump operators
    Tpts1: numbers of points to sample along dimension 1
    Tpst2: number of points to sample along dimension 2
    tmix: mixing time during the NOESY protocol
    dt1: delta of time to sample along the first dimension
    dt2: delta of time to sample along the second dimension
    zerofill1: 
    zerofill2:
    Lx: the total angular component along x for the system
    Ly: the total angular component along y for the system
    """

    if len(JumpOps)>1:
        print("For now, considering only one jump operator")
        exit()

    sys_qubs = of.count_qubits(Ham)
    
    spJumpOp = of.get_sparse_operator(JumpOps[0],n_qubits=sys_qubs)
    EmbJump = np.kron(spJumpOp.toarray(),np.array([[0,1],[0,0]]))
    EmbJump+= EmbJump.conj().T

    sp_Ham = of.get_sparse_operator(Ham,n_qubits=(sys_qubs+1)) #Needs to be modified when considering more ancillary qubits

    Lnet = sp_Ham.toarray()+EmbJump
    DilHam_1 = expm(-1j*Lnet*dt1)
    DilHam_2 = expm(-1j*Lnet*dt2)
    #pulse_mix = expm(-1j*Lnet*tmix)
    sp_Lx = of.get_sparse_operator(Lx,n_qubits=(sys_qubs+1))
    sp_Ly = of.get_sparse_operator(Ly,n_qubits=(sys_qubs+1))

    pulse_90x = expm(-1j*sp_Lx.toarray()*np.pi/2)
    pulse_90y = expm(-1j*sp_Ly.toarray()*np.pi/2)
    pulse_90mx = expm(1j*sp_Lx.toarray()*np.pi/2)
    pulse_90my = expm(1j*sp_Ly.toarray()*np.pi/2)


    FID_1 = np.zeros([Tpts2,Tpts1],dtype=complex)
    FID_2 = np.zeros([Tpts2,Tpts1],dtype=complex)
    FID_3 = np.zeros([Tpts2,Tpts1],dtype=complex)
    FID_4 = np.zeros([Tpts2,Tpts1],dtype=complex)

    #Build the circuits that are going to be re-used during the simulation...
    anc_qubs = 1 
    allqubits = cirq.LineQubit.range(sys_qubs+anc_qubs)
    anc_register = allqubits[sys_qubs:(sys_qubs+anc_qubs)]
    #generation of constituent circuits...
    
    #TrotCirc1 = TrotStepLinb(DilHam_1,allqubits,anc_register) 
    #TrotCirc2 = TrotStepLinb(DilHam_2,allqubits,anc_register)

    CircPulse_90x = BuildUCirc(pulse_90x,allqubits)
    CircPulse_90y = BuildUCirc(pulse_90y,allqubits)
    CircPulse_90mx = BuildUCirc(pulse_90mx,allqubits)
    CircPulse_90my = BuildUCirc(pulse_90my,allqubits)
    #CircPulse3 =  BuildUCirc(pulse_90y,allqubits)


    tmix_steps = int(tmix/dt1)

    #We need to loop over different state-preparation circuits, but for the two-spin system, we need to consider the only product
    #state with positive total magnetization eigenvalue?

    CircInit = BuildUCirc(np.eye(2**(sys_qubs+anc_qubs)),allqubits)

    #rho_stack1_1.append(pulse_90y@pulse_mix@pulse_90x@rho_stack[i])
    #rho_stack1_2.append(pulse_90y@pulse_mix@pulse_90y@rho_stack[i])
    #rho_stack1_3.append(pulse_90y@pulse_mix@pulse_90mx@rho_stack[i])
    #rho_stack1_4.append(pulse_90y@pulse_mix@pulse_90my@rho_stack[i])
    #Tot_points = Tpts1*Tpts2

    #for the purpose of tracking progress in the simulation, we report every "stride" size, to have 5 reports during the whole simulation
    stride = int(Tpts1/5)
    stride_count = 0
    for t1_steps in range(Tpts1):
        for t2_steps in range(Tpts2):
            #We generate 4 circuits for the different pulse sequences...
            #Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit,TrotCirc1,TrotCirc2,CircPulse1,CircPulse2,Circpulse3)

            CircExp1_X, CircExp1_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit,DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90x,CircPulse_90y,allqubits,anc_register)

            CircExp2_X, CircExp2_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit,DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90y,CircPulse_90y,allqubits,anc_register)

            CircExp3_X, CircExp3_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit,DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90mx,CircPulse_90y,allqubits,anc_register)

            CircExp4_X, CircExp4_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit,DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90my,CircPulse_90y,allqubits,anc_register)

            #TODO: use Meas_TotMag(Circ,qubits,nqubs,samples) to compute total magnetization for each of the circuits and store in array.
            ExpcMag1_X = Meas_TotMag(CircExp1_X,allqubits,sys_qubs,samples)
            ExpcMag1_Y = Meas_TotMag(CircExp1_Y,allqubits,sys_qubs,samples)

            ExpcMag2_X = Meas_TotMag(CircExp2_X,allqubits,sys_qubs,samples)
            ExpcMag2_Y = Meas_TotMag(CircExp2_Y,allqubits,sys_qubs,samples)

            ExpcMag3_X = Meas_TotMag(CircExp3_X,allqubits,sys_qubs,samples)
            ExpcMag3_Y = Meas_TotMag(CircExp3_Y,allqubits,sys_qubs,samples)

            ExpcMag4_X = Meas_TotMag(CircExp4_X,allqubits,sys_qubs,samples)
            ExpcMag4_Y = Meas_TotMag(CircExp4_Y,allqubits,sys_qubs,samples)

            FID_1[t2_steps,t1_steps] = ExpcMag1_X-1j*ExpcMag1_Y
            FID_2[t2_steps,t1_steps] = ExpcMag2_X-1j*ExpcMag2_Y
            FID_3[t2_steps,t1_steps] = ExpcMag3_X-1j*ExpcMag3_Y
            FID_4[t2_steps,t1_steps] = ExpcMag4_X-1j*ExpcMag4_Y
        
        if t1_steps%stride==0:
            stride_count+=1
            print("Finished processing ",stride_count*20,"% of total simulation" )
        
            
    return FID_1, FID_2, FID_3, FID_4


#####For verification of circuit-based simulations with density matrix simualtions...
#Modification of the simulation of circuits and estimation of expectation values...

def Meas_TotMagDen_Mat(Circ,qubits,nqubs,samples):
    """ 
    More efficient estimation of total magnetization than that in Meas_TotMag. We measure simultaneously all system qubits, and post-process the results for estimation of total magnetization
    instead of using different versions of the circuit for estimation of each local observable. This can be done due to the shared eigenbasis of the total magnetization and Z0Z1...Z_{N-1}
    """
    simulator = cirq.DensityMatrixSimulator()

    tot_nqubs = len(qubits)
    #for i in range(nqubs):
    #    Circ.append(cirq.measure(qubits[i],key='sys'+str(i)))
    TotMag = of.QubitOperator()
    for i in range(nqubs):
        TotMag+=of.QubitOperator('Z'+str(i))
    
    TotMag_Mat = of.get_sparse_operator(TotMag,n_qubits=tot_nqubs) 
    TotMag_Mat = TotMag_Mat.toarray()

    Expect_Zs = []

    for i in range(samples):

        result = simulator.simulate(Circ)

        den_mat = result.final_density_matrix

        Expect_Zs.append(np.trace(TotMag_Mat@den_mat))
        
    
    return np.sum(Expect_Zs)/samples


def CircGen2D_FID_DenMat(Ham,JumpOps,Tpts1,Tpts2,tmix,dt1,dt2,Lx,Ly,samples=10):
    """
    Function to sample the 2D FID signals needed for calculation of 2D spectrum in a NOESY experiment.
    Args:
    Ham: the Hamiltonian in OpenFermion format to be simulated
    JumpOps: list of OpenFermion jump operators
    Tpts1: numbers of points to sample along dimension 1
    Tpst2: number of points to sample along dimension 2
    tmix: mixing time during the NOESY protocol
    dt1: delta of time to sample along the first dimension
    dt2: delta of time to sample along the second dimension
    zerofill1: 
    zerofill2:
    Lx: the total angular component along x for the system
    Ly: the total angular component along y for the system
    """

    if len(JumpOps)>1:
        print("For now, considering only one jump operator")
        exit()

    sys_qubs = of.count_qubits(Ham)
    
    spJumpOp = of.get_sparse_operator(JumpOps[0],n_qubits=sys_qubs)
    EmbJump = np.kron(spJumpOp.toarray(),np.array([[0,1],[0,0]]))
    EmbJump+= EmbJump.conj().T

    sp_Ham = of.get_sparse_operator(Ham*S_plus(sys_qubs)*S_minus(sys_qubs),n_qubits=(sys_qubs+1)) #Needs to be modified when considering more ancillary qubits

    Lnet1 = sp_Ham.toarray()*np.sqrt(dt1)+EmbJump
    Lnet2 = sp_Ham.toarray()*np.sqrt(dt2)+EmbJump
    DilHam_1 = expm(-1j*Lnet1*np.sqrt(dt1))
    DilHam_2 = expm(-1j*Lnet2*np.sqrt(dt2))
    #pulse_mix = expm(-1j*Lnet*tmix)
    sp_Lx = of.get_sparse_operator(Lx,n_qubits=(sys_qubs+1))
    sp_Ly = of.get_sparse_operator(Ly,n_qubits=(sys_qubs+1))

    pulse_90x = expm(-1j*sp_Lx.toarray()*np.pi/2)
    pulse_90y = expm(-1j*sp_Ly.toarray()*np.pi/2)
    pulse_90mx = expm(1j*sp_Lx.toarray()*np.pi/2)
    pulse_90my = expm(1j*sp_Ly.toarray()*np.pi/2)


    FID_1 = np.zeros([Tpts2,Tpts1],dtype=complex)
    FID_2 = np.zeros([Tpts2,Tpts1],dtype=complex)
    FID_3 = np.zeros([Tpts2,Tpts1],dtype=complex)
    FID_4 = np.zeros([Tpts2,Tpts1],dtype=complex)

    #Build the circuits that are going to be re-used during the simulation...
    anc_qubs = 1 
    allqubits = cirq.LineQubit.range(sys_qubs+anc_qubs)
    anc_register = allqubits[sys_qubs:(sys_qubs+anc_qubs)]
    #generation of constituent circuits...
    
    #TrotCirc1 = TrotStepLinb(DilHam_1,allqubits,anc_register) 
    #TrotCirc2 = TrotStepLinb(DilHam_2,allqubits,anc_register)

    CircPulse_90x = BuildUCirc(pulse_90x,allqubits)
    CircPulse_90y = BuildUCirc(pulse_90y,allqubits)
    CircPulse_90mx = BuildUCirc(pulse_90mx,allqubits)
    CircPulse_90my = BuildUCirc(pulse_90my,allqubits)
    #CircPulse3 =  BuildUCirc(pulse_90y,allqubits)


    tmix_steps = int(tmix/dt1)
    print("Number of time steps for mixing time:", tmix_steps)
    #We need to loop over different state-preparation circuits, but for the two-spin system, we need to consider the only product
    #state with positive total magnetization eigenvalue?

    CircInit1 = BuildUCirc(np.eye(2**(sys_qubs+anc_qubs)),allqubits)
    temp_op = QubitOperator('X0 X1')

    CircInit2 = BuildUCirc(of.get_sparse_operator(temp_op,n_qubits=(sys_qubs+1)).toarray(),allqubits)

    CircInit_array = [CircInit1,CircInit2]

    #rho_stack1_1.append(pulse_90y@pulse_mix@pulse_90x@rho_stack[i])
    #rho_stack1_2.append(pulse_90y@pulse_mix@pulse_90y@rho_stack[i])
    #rho_stack1_3.append(pulse_90y@pulse_mix@pulse_90mx@rho_stack[i])
    #rho_stack1_4.append(pulse_90y@pulse_mix@pulse_90my@rho_stack[i])
    #Tot_points = Tpts1*Tpts2

    #for the purpose of tracking progress in the simulation, we report every "stride" size, to have 5 reports during the whole simulation
    stride = int(Tpts1/5)
    stride_count = 0
    
    for t1_steps in range(Tpts1):
        for t2_steps in range(Tpts2):
            #We generate 4 circuits for the different pulse sequences...
            #Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit,TrotCirc1,TrotCirc2,CircPulse1,CircPulse2,Circpulse3)

            for i in range(len(CircInit_array)):

                CircExp1_X, CircExp1_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit_array[i],DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90x,CircPulse_90y,allqubits,anc_register)

                CircExp2_X, CircExp2_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit_array[i],DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90y,CircPulse_90y,allqubits,anc_register)

                CircExp3_X, CircExp3_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit_array[i],DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90mx,CircPulse_90y,allqubits,anc_register)

                CircExp4_X, CircExp4_Y = Build2D_circ(t1_steps,t2_steps,tmix_steps,CircInit_array[i],DilHam_1,DilHam_2,CircPulse_90x,CircPulse_90my,CircPulse_90y,allqubits,anc_register)

                #TODO: use Meas_TotMag(Circ,qubits,nqubs,samples) to compute total magnetization for each of the circuits and store in array.
                #Meas_TotMagDen_Mat(Circ,qubits,nqubs,samples)
                ExpcMag1_X = Meas_TotMagDen_Mat(CircExp1_X,allqubits,sys_qubs,samples)
                ExpcMag1_Y = Meas_TotMagDen_Mat(CircExp1_Y,allqubits,sys_qubs,samples)

                ExpcMag2_X = Meas_TotMagDen_Mat(CircExp2_X,allqubits,sys_qubs,samples)
                ExpcMag2_Y = Meas_TotMagDen_Mat(CircExp2_Y,allqubits,sys_qubs,samples)

                ExpcMag3_X = Meas_TotMagDen_Mat(CircExp3_X,allqubits,sys_qubs,samples)
                ExpcMag3_Y = Meas_TotMagDen_Mat(CircExp3_Y,allqubits,sys_qubs,samples)

                ExpcMag4_X = Meas_TotMagDen_Mat(CircExp4_X,allqubits,sys_qubs,samples)
                ExpcMag4_Y = Meas_TotMagDen_Mat(CircExp4_Y,allqubits,sys_qubs,samples)

                FID_1[t2_steps,t1_steps] += (-1)**i * (ExpcMag1_X-1j*ExpcMag1_Y)
                FID_2[t2_steps,t1_steps] += (-1)**i * (ExpcMag2_X-1j*ExpcMag2_Y)
                FID_3[t2_steps,t1_steps] += (-1)**i * (ExpcMag3_X-1j*ExpcMag3_Y)
                FID_4[t2_steps,t1_steps] += (-1)**i * (ExpcMag4_X-1j*ExpcMag4_Y)
            
        if t1_steps%stride==0:
            stride_count+=1
            print("Finished processing ",stride_count*20,"% of total simulation" )
    
        

            
    return FID_1, FID_2, FID_3, FID_4


# Function to calculate the partial trace
def partial_trace(rho, keep_qubits, dims):
    """
    Compute the partial trace of the density matrix `rho`
    over the qubits that are not in `keep_qubits`.
    - `rho`: The full density matrix.
    - `keep_qubits`: List of qubits to keep after tracing out others.
    - `dims`: List of dimensions for each qubit in the system.
    """
    # Reshape the density matrix into 4D array (for 2-qubit system: (2, 2, 2, 2))
    reshaped_rho = np.reshape(rho, dims + dims)
    
    # Trace out the qubits not in keep_qubits
    for i in reversed(range(len(dims))):
        if i not in keep_qubits:
            reshaped_rho = np.trace(reshaped_rho, axis1=i, axis2=i + len(dims))
    
    # Reshape the result back to a 2D array for the reduced density matrix
    keep_dims = [dims[i] for i in keep_qubits]
    return np.reshape(reshaped_rho, (np.prod(keep_dims), np.prod(keep_dims)))













