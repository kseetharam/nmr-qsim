import openfermion as of

import sys
sys.path.append('./utils/')

#from 
from scipy.linalg import expm
import cirq
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
import pickle

from multiprocessing import Pool
import numpy as np
from scipy.linalg import expm


def Normalize_and_weightOps(ListOps):
    """
    ListOps contains the generators of evolution, either the representation of the coherent part of evolution or the jump operators 
    """
    List_weights=[]

    for i in range(len(ListOps)):
        List_weights.append(np.max(np.abs(np.linalg.eigvals(ListOps[i]))))

    #Normalize the operator according to the weights...
    Norm_ops =[]

    for i in range(len(ListOps)):
        Norm_ops.append(ListOps[i]/List_weights[i])

    return Norm_ops, List_weights

#####Rewriting, cleaning and tailoring some functions for QDrift simulation of the coherent part of the Hamiltonian
def BuildCohQDriftChann_fromOps(time_evol,Norm_ops,pks,Gamma):
    """
    It is assumed that all operators in Norm_ops corresponds to fragments of the coherent part of the Liouvillian 
    """
    Qdrift_chann = pks[0]*expm(-1j*Norm_ops[0]*time_evol*Gamma)
    for i in range(1,len(pks)):
        Qdrift_chann += pks[i]*expm(-1j*Norm_ops[i]*time_evol*Gamma)

    return Qdrift_chann

def CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,time_evol,Nsteps,rho):
    
    rho_c = np.copy(rho)

    if time_evol==0:
        chann = np.eye(rho_c.shape[0])
    else:
        chann= BuildCohQDriftChann_fromOps(time_evol/Nsteps,Norm_ops,pks,Gamma)

    for i in range(Nsteps):
        rho_c = chann@rho_c

    return rho_c

def FieldGrad(Npts_grad,rho_stack,dim,Lz):
    """ 
    Inclusion of magnetic field gradient in the simulation. Warning: it modifies and returns the rho_stack array
    """

    phi = np.linspace(0,2*np.pi,Npts_grad)

    gradmat = np.zeros([dim,dim,Npts_grad],dtype=complex)

    for k in range(Npts_grad):
        gradmat[:,:,k] = expm(-1j*Lz*phi[k])


    for j in range(len(rho_stack)): #iterating over the number of time points....
        rho_temp = rho_stack[j]
        rho_sum = 0
        for i in range(Npts_grad):
            rho_sum = rho_sum + gradmat[:,:,i]@rho_temp

        rho_sum = rho_sum/Npts_grad
        rho_stack[j] = rho_sum


    return rho_stack

def get_last_QDriftJump(List_gens,time_evol,NSteps,U):
    """
    Calculation of an effective QDrift channel. Based on the observation that the action of jump operators are in essence rare events 
    """
    Norm_ops, List_weights = Normalize_and_weightOps(List_gens)

    List_weights = np.array(List_weights)
    Gamma =  np.sum(List_weights)

    pks = (1.0/Gamma)*List_weights

    #Eff_probs = np.zeros(len(List_gens))

    #Eff_probs[0] = pks[0]**(NSteps) #probability of sampling coherent evolution only

    ###We define this channel as the product of two channels: the right one applies the jump operator before the action of the unitary pulse U
    omega = time_evol*Gamma/NSteps
    Left_chan = pks[0]*U@expm(-1j*Norm_ops[0]*omega)
    for k in range(1,len(pks)):
        Left_chan+=NSteps*pks[k]*expm(omega*U@Norm_ops[k]@np.conjugate(np.transpose(U)))@U

    ##right channel...
    #Right_chan = pks[0]**(NSteps-1)*expm(-1j*Norm_ops[0]*(NSteps-1)*omega)


    return pks[0]**(NSteps-1)*Left_chan


def get_eff_tmixpulse(List_gens,tmix,Ntmix):
    """
    With the aim of removing the mixing time in a NOESY protocol, we effectively substitute it by a QDrift channel that consist of a single time step,
    the probabilities and the time step of this channel are calculated assuming that we are effectively substituting the channel by the composition of Ntmix qdrift channels.
    Args:
    List_gens: list of the generators of Liouvillian evolution, it is assumed that the first element corresponds to the coherent part of the Liovillian whereas the Jump
    operators correspond to the rest of the indices
    tmix: the mixng time
    Ntmix: the number of time steps assumed to be effectively simulated during the mixing time
    """
    Norm_ops, List_weights = Normalize_and_weightOps(List_gens)

    List_weights = np.array(List_weights)
    Gamma =  np.sum(List_weights)

    pks = (1.0/Gamma)*List_weights

    #Eff_probs = np.zeros(len(List_gens))

    #Eff_probs[0] = pks[0]**(NSteps) #probability of sampling coherent evolution only

    ###We define this channel as the product of two channels: the right one applies the jump operator before the action of the unitary pulse U
    omega = tmix*Gamma/Ntmix

    Eff_chan = pks[0]*expm(-1j*Norm_ops[0]*omega)
    for k in range(1,len(pks)):
        Eff_chan+=Ntmix*pks[k]*expm(omega*Norm_ops[k])


    return pks[0]**(Ntmix-1)*Eff_chan








def GenFID_SingJump_noTmix_GradField_SetMaxDisc(Ham,IntOps,JumpOps,T1,T2,rho0,coil,tmix,dt1,dt2,Ntmix,Lx,Ly,Lz,Nmax):
    """
    We aim to perform the simulation by setting a maximum number of samples from a QDrift channel to perform the simulation of t1 and t2 timescales
    to perform the simulation of NOESY spectrum. 
    Args:
    Ham: the Zeeman Hamiltonian 
    IntOps: the list of Hamiltonian fragments, such that their addition to the Zeeaman Hamiltonian define the coherent contribution to the Liovillian
    JumOps: List of jump operators to include in the simulation
    T1: total simulation time along the t1 axis
    T2: total simulation time along the t2 axis
    rho0: initial density matrix
    coil: the observable to measure 
    tmix: the mixing time
    dt1: the time step that defines the time-resolution of the grid along the t1 axis 
    dt2: the time step that defines the time resolution of the grid along the t2 axis
    Ntmix: the number of samples assumed to be taken during the mixing time through a QDroft channel. Note that in practice this channel is approximated as an effective one where
    the mixing time is removed
    Lx,Ly,Lz: collective angular momentum operators
    Nmax: maximum number of Qdrift samples for the simulation of t1,t2 times (i.e. the maximum number of samples is fixed to 2*Nmax for time propagation of t1+t2)
    """
    rho0 = rho0.flatten()

    Npts_grad = 100
    dim = rho0.shape[0]
    

    Tpts1 = int(np.floor(T1/dt1))
    Tpts2 = int(np.floor(T2/dt2))

    print("Number of points for T1",Tpts1)
    print("Number of points for T2",Tpts2)
    #print("Number of Trotter steps for QDrift simulation of tmix", Ntmix)
    

    pulse_90x = expm(-1j*Lx*np.pi/2)
    pulse_90y = expm(-1j*Ly*np.pi/2)
    pulse_90mx = expm(1j*Lx*np.pi/2)
    pulse_90my = expm(1j*Ly*np.pi/2)

    #First 90x pulse:
    rho_t = np.copy(rho0)
    rho_t = np.dot(pulse_90x,rho_t)

    rho_stack = []
    rho_stack.append(rho_t)

    ###Getting the list of operators and probabilities for the QDrift channel...
    Norm_ops, List_weights = Normalize_and_weightOps([Ham]+IntOps)

    List_weights = np.array(List_weights)
    Gamma =  np.sum(List_weights)

    pks = (1.0/Gamma)*List_weights

    #QDriftEvol_fromNormOps(Norm_ops,pks,Gamma,time_evol,Nsteps,rho)

    #rho_temp = np.copy(rho_t)
    for i in range(1,Tpts1):
        ####after the number of time points exceeds the maximum, we keep the number of samples in the QDrift channel fixed...
        if i > Nmax:
            #rho_t = QDriftEvol([Ham]+JumpOps,i*dt1,Nmax,rho0)
            rho_t = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,i*dt1,Nmax,rho0)

            #rho_temp = np.dot(L_dt1,rho_temp)
            rho_stack.append(rho_t)

        else:
            #rho_t = QDriftEvol([Ham]+JumpOps,i*dt1,i,rho0)
            rho_t = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,i*dt1,i,rho0)

            #rho_temp = np.dot(L_dt1,rho_temp)
            rho_stack.append(rho_t)


    rho_stack1_1 = []
    rho_stack1_2 = []
    rho_stack1_3 = []
    rho_stack1_4 = []


    Hcoh = Ham+sum(IntOps)
    eff_pulse_90y = get_last_QDriftJump([Hcoh]+JumpOps,tmix,Ntmix,pulse_90y)
    

    ###Aplication of second 90 deg pulse

    print("Application of second 90 deg pulse")
    for i in range(Tpts1):
        rho_stack1_1.append(pulse_90x@rho_stack[i])
        rho_stack1_2.append(pulse_90y@rho_stack[i])
        rho_stack1_3.append(pulse_90mx@rho_stack[i])
        rho_stack1_4.append(pulse_90my@rho_stack[i])


    #####Aplication of gradient...
    rho_stack1_1 = FieldGrad(Npts_grad,rho_stack1_1,dim,Lz)
    rho_stack1_2 = FieldGrad(Npts_grad,rho_stack1_2,dim,Lz)
    rho_stack1_3 = FieldGrad(Npts_grad,rho_stack1_3,dim,Lz)
    rho_stack1_4 = FieldGrad(Npts_grad,rho_stack1_4,dim,Lz)

    print("Application of third 90 deg pulse")
    #application of third pulse...
    for i in range(Tpts1):
        rho_stack1_1[i] = eff_pulse_90y@rho_stack1_1[i]
        rho_stack1_2[i] = eff_pulse_90y@rho_stack1_2[i]
        rho_stack1_3[i] = eff_pulse_90y@rho_stack1_3[i]
        rho_stack1_4[i] = eff_pulse_90y@rho_stack1_4[i]



    fid_temp_1 = np.zeros([Tpts2,Tpts1],dtype=complex)
    fid_temp_2 = np.zeros([Tpts2,Tpts1],dtype=complex)
    fid_temp_3 = np.zeros([Tpts2,Tpts1],dtype=complex)
    fid_temp_4 = np.zeros([Tpts2,Tpts1],dtype=complex)

    
    for i in range(Tpts1):
        rho1 = np.copy(rho_stack1_1[i])
        rho2 = np.copy(rho_stack1_2[i])
        rho3 = np.copy(rho_stack1_3[i])
        rho4 = np.copy(rho_stack1_4[i])

        rho1_t2 = np.copy(rho1)
        rho2_t2 = np.copy(rho2)
        rho3_t2 = np.copy(rho3)
        rho4_t2 = np.copy(rho4)

        for j in range(Tpts2):
            fid_temp_1[j,i] = np.dot(coil,rho1_t2)

            if j > Nmax:
                #rho1_t2 = QDriftEvol([Ham]+JumpOps,j*dt2,Nmax,rho1)
                rho1_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,Nmax,rho1)
            else:
                rho1_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,j,rho1)
            
            #rho1 = L_dt2@rho1

            fid_temp_2[j,i] = np.dot(coil,rho2_t2)
            if j > Nmax:
                rho2_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,Nmax,rho2)
            else:
                rho2_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,j,rho2)
            #rho2 = L_dt2@rho2

            fid_temp_3[j,i] = np.dot(coil,rho3_t2)
            if j > Nmax:
                rho3_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,Nmax,rho3)
            else:
                rho3_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,j,rho3)

            #rho3 = L_dt2@rho3

            fid_temp_4[j,i] = np.dot(coil,rho4_t2)
            if j > Nmax:
                rho4_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,Nmax,rho4)
            else:
                rho4_t2 = CohQDriftEvol_fromNormOps(Norm_ops,pks,Gamma,j*dt2,j,rho4)

            #rho4 = L_dt2@rho4
        print("Finishing iteration: ",i)
    
    return fid_temp_1, fid_temp_2, fid_temp_3, fid_temp_4


def compute_fid_for_t1(args):
    """
    Computes the FID for a single t1 point.
    Args:
        args: A tuple containing the parameters:
            - i: Index of the t1 point
            - rho_stack: List of prepared states for t1 points
            - Npts_grad: Number of gradient points
            - dim: Dimension of the density matrix
            - Lz: Angular momentum operator in z direction
            - eff_pulse_90y: Effective pulse matrix
            - coil: Observable matrix
            - Norm_ops: Normalized operators for QDrift
            - pks: Probabilities for QDrift
            - Gamma: Normalization constant for QDrift
            - Tpts2: Number of t2 points
            - dt2: Time step along t2
            - Nmax: Maximum number of QDrift samples
    Returns:
        Tuple of FID arrays (fid1, fid2, fid3, fid4) for the given t1 point.
    """
    (i, rho_stack1, rho_stack2, rho_stack3, rho_stack4, Npts_grad, dim, Lz, pulse_90y, coil, Tpts2, L_dt2) = args

    # Extract prepared states for this t1 step
    rho_t1_1 = pulse_90y @ FieldGrad(Npts_grad, [rho_stack1[i]], dim, Lz)[0]
    rho_t1_2 = pulse_90y @ FieldGrad(Npts_grad, [rho_stack2[i]], dim, Lz)[0]
    rho_t1_3 = pulse_90y @ FieldGrad(Npts_grad, [rho_stack3[i]], dim, Lz)[0]
    rho_t1_4 = pulse_90y @ FieldGrad(Npts_grad, [rho_stack4[i]], dim, Lz)[0]
    

    #rho1 = np.copy(rho_t1_1)
    #rho2 = np.copy(rho_t1_2)
    #rho3 = np.copy(rho_t1_3)
    #rho4 = np.copy(rho_t1_4)

    # Initialize arrays for fid computations
    fid1, fid2, fid3, fid4 = np.zeros(Tpts2, dtype=complex), np.zeros(Tpts2, dtype=complex), np.zeros(Tpts2, dtype=complex), np.zeros(Tpts2, dtype=complex)
    
    for j in range(Tpts2):
        #samples = Nmax if j > Nmax else j
        fid1[j] = np.dot(coil, rho_t1_1)
        rho_t1_1 = L_dt2@rho_t1_1 # CohQDriftEvol_fromNormOps(Norm_ops, pks, Gamma, j * dt2, samples, rho1)
        
        fid2[j] = np.dot(coil, rho_t1_2)
        rho_t1_2 = L_dt2@rho_t1_2 # CohQDriftEvol_fromNormOps(Norm_ops, pks, Gamma, j * dt2, samples, rho2)
        
        fid3[j] = np.dot(coil, rho_t1_3)
        rho_t1_3 = L_dt2@rho_t1_3 #CohQDriftEvol_fromNormOps(Norm_ops, pks, Gamma, j * dt2, samples, rho3)
        
        fid4[j] = np.dot(coil, rho_t1_4)
        rho_t1_4 = L_dt2@rho_t1_4 #CohQDriftEvol_fromNormOps(Norm_ops, pks, Gamma, j * dt2, samples, rho4)
    
    return fid1, fid2, fid3, fid4


def GenFID_SingJump_noTmix_GradField_MaxQD_Parallel(Ham, IntOps, JumpOps, T1, T2, rho0, coil, tmix, dt1, dt2, Ntmix, Lx, Ly, Lz):
    rho0 = rho0.flatten()
    Npts_grad = 100
    dim = rho0.shape[0]
    
    Tpts1 = int(np.floor(T1 / dt1))
    Tpts2 = int(np.floor(T2 / dt2))

    print("Number of points for T1:", Tpts1)
    print("Number of points for T2:", Tpts2)
    
    pulse_90x = expm(-1j * Lx * np.pi / 2)
    pulse_90y = expm(-1j * Ly * np.pi / 2)
    pulse_90mx = expm(1j * Lx * np.pi / 2)
    pulse_90my = expm(1j * Ly * np.pi / 2)
    
    rho_init = np.dot(pulse_90x, np.copy(rho0))
    rho_stack = [rho_init]

    Norm_ops, List_weights = Normalize_and_weightOps([Ham] + IntOps)
    List_weights = np.array(List_weights)
    Gamma = np.sum(List_weights)
    pks = (1.0 / Gamma) * List_weights

    L_dt1 = BuildCohQDriftChann_fromOps(dt1,Norm_ops,pks,Gamma)
    
    for i in range(1, Tpts1):
        #samples = Nmax if i > Nmax else i
        rho_init = L_dt1@rho_init
        rho_stack.append(rho_init)
    
    Hcoh = Ham + sum(IntOps)
    #eff_pulse_90y = get_last_QDriftJump([Hcoh] + JumpOps, tmix, Ntmix, pulse_90y)
    eff_mix_pulse = get_eff_tmixpulse([Hcoh] + JumpOps,tmix,Ntmix)
    

    rho_stack1_1 = []
    rho_stack1_2 = []
    rho_stack1_3 = []
    rho_stack1_4 = []

    print("Application of second 90 deg pulse")
    for i in range(Tpts1):
        rho_stack1_1.append(pulse_90x@rho_stack[i])
        rho_stack1_2.append(pulse_90y@rho_stack[i])
        rho_stack1_3.append(pulse_90mx@rho_stack[i])
        rho_stack1_4.append(pulse_90my@rho_stack[i])


    #####Aplication of gradient...
    rho_stack1_1 = FieldGrad(Npts_grad,rho_stack1_1,dim,Lz)
    rho_stack1_2 = FieldGrad(Npts_grad,rho_stack1_2,dim,Lz)
    rho_stack1_3 = FieldGrad(Npts_grad,rho_stack1_3,dim,Lz)
    rho_stack1_4 = FieldGrad(Npts_grad,rho_stack1_4,dim,Lz)

    ###Application of effective mixing time channel...
    for i in range(Tpts1):
        rho_stack1_1[i] = eff_mix_pulse@rho_stack1_1[i]
        rho_stack1_2[i] = eff_mix_pulse@rho_stack1_2[i]
        rho_stack1_3[i] = eff_mix_pulse@rho_stack1_3[i]
        rho_stack1_4[i] = eff_mix_pulse@rho_stack1_4[i]


    Norm_ops, List_weights = Normalize_and_weightOps([Ham] + IntOps)
    List_weights = np.array(List_weights)
    Gamma = np.sum(List_weights)
    pks = (1.0 / Gamma) * List_weights

    L_dt2 = BuildCohQDriftChann_fromOps(dt2,Norm_ops,pks,Gamma)

    #i, rho_stack1, rho_stack2, rho_stack3, rho_stack4, Npts_grad, dim, Lz, eff_pulse_90y, coil, Norm_ops, pks, Gamma, Tpts2, dt2, Nmax
    # Prepare arguments for parallel processing
    args = [
        (
            i, rho_stack1_1,rho_stack1_2, rho_stack1_3, rho_stack1_4, Npts_grad, dim, Lz, pulse_90y, coil, 
            Tpts2, L_dt2
        )
        for i in range(Tpts1)
    ]
    
    print("Starting parallel execution....")
    # Parallelize the t1 loop
    with Pool() as pool:
        results = pool.map(compute_fid_for_t1, args)
    
    # Combine results
    fid_temp_1 = np.zeros((Tpts2, Tpts1), dtype=complex)
    fid_temp_2 = np.zeros((Tpts2, Tpts1), dtype=complex)
    fid_temp_3 = np.zeros((Tpts2, Tpts1), dtype=complex)
    fid_temp_4 = np.zeros((Tpts2, Tpts1), dtype=complex)
    
    for i, (f1, f2, f3, f4) in enumerate(results):
        fid_temp_1[:, i] = f1
        fid_temp_2[:, i] = f2
        fid_temp_3[:, i] = f3
        fid_temp_4[:, i] = f4
    
    return fid_temp_1, fid_temp_2, fid_temp_3, fid_temp_4



#Nmax = int(sys.argv[1])

#print("The maximal number of samples considered in the QDrift channel for the simulation of each time axis is: ", Nmax)

###load all the required parameters...
f =open('./data/ALA_expGradField_AllParams.pk','rb')

params = pickle.load(f)

Zeem_Ham = params['ZeemanH']
ListInts = params['ListInts']
JumpOps = params['JumpOps']
T1 = params['T1']
T2 = params['T2']
rho0 = params['rho0']
coil = params['coil']
tmix = params['tmix']
dt1 = params['dt1']
dt2 = params['dt2']
Lx = params['Lx']
Ly = params['Ly']
Lz = params['Lz']

###Given the time needed to complete the calculations, we further reduce the number of time points along t1 and t2...
#dt1 = T1/10240
#dt2 = T2/10240


Ntmix = 2631

if __name__=="__main__":
    
    fid_temp_1, fid_temp_2, fid_temp_3, fid_temp_4 = GenFID_SingJump_noTmix_GradField_MaxQD_Parallel(Zeem_Ham, ListInts, JumpOps, T1, T2, rho0, coil, tmix, dt1, dt2, Ntmix, Lx, Ly, Lz)

    DictFID = {'fid1': fid_temp_1, 'fid2': fid_temp_2, 'fid3':fid_temp_3, 'fid4': fid_temp_4}

    with open('./data/ALA_SimplifiedFIDParallelmaxNQD.pk', 'wb') as handle:
        pickle.dump(DictFID, handle)








