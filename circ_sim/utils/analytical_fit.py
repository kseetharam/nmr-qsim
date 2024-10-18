import numpy as np
import sys
sys.path.append('./')
import pickle

from basis_utils import MatRepLib, InnProd, Linb_Channel
from basis_utils import Sz,S_plus,S_minus

from concurrent.futures import ProcessPoolExecutor


def Dip_tensor(coord1,coord2):
    diff = coord2-coord1
    r = np.sqrt(np.dot(diff,diff))

    A = np.zeros([3,3])

    for i in range(3):
        for j in range(3):
            #A[i,j]=3.0*diff[i]*diff[j]/r**2
            #if i==j:
            #    A[i,j]+=-1.0
            if i==j:
                A[i,j]=3.0*diff[i]*diff[j]/r**2-1.0
            else:
                A[i,j]=1.5*diff[i]*diff[j]/r**2
                

    return A

def get_sym_as(Atens):
    a_s = np.zeros(5,dtype=complex)
    a_s[0] = 0.5*(Atens[0,0]-Atens[1,1]+1j*(Atens[0,1]+Atens[1,0])) #m=-2
    a_s[4] = np.conjugate(a_s[0])                                   #m=2
    a_s[1] = 0.5*(Atens[0,2]+Atens[2,0]+1j*(Atens[1,2]+Atens[2,1])) #m = -1
    a_s[3] = -1.0*np.conjugate(a_s[1])                              #m=1
    a_s[2] = np.sqrt(1.0/6.0)*(2*Atens[2,2]-(Atens[1,1]+Atens[2,2])) #m=0

    return a_s


def SpecFunc(w,tc):

    return 0.2*tc/(1 + w**2 * tc**2)


def GammaRates(w,tc,coord1,coord2,coord3,coord4,gamma,iso_av=True):
    """
    Function to compute the damping constants for Linbaldians.
    """
    hbar = 1.054571628*1e-34
    diff1 = coord2-coord1
    diff2 = coord4 - coord3
    r1 = np.sqrt(np.dot(diff1,diff1))
    r2 = np.sqrt(np.dot(diff2,diff2))

    if iso_av:
        x1 = diff1[0]
        y1 = diff1[1]
        z1 = diff1[2]
    
        x2 = diff2[0]
        y2 = diff2[1]
        z2 = diff2[2]
    
        r1 = np.sqrt(x1**2+y1**2+z1**2)
        r2 = np.sqrt(x2**2+y2**2+z2**2)
    
        Invariant = y2**2 * (2*y1**2-z1**2)-x2**2 * (y1**2+z1**2)+6*y1*y2*z1*z2-(y1**2-2*z1**2)*z2**2
        Invariant += 6*x1*x2*(y1*y2+z1*z2)+x1**2 * (2*x2**2-y2**2-z2**2)
        Invariant = (3/(r1**2 * r2**2))*Invariant
        
        Apref = Invariant
        #print("prefactor is", Apref)
        
    else:
        dip_ten1 = Dip_tensor(coord1,coord2)
        a_s1 = get_sym_as(dip_ten1)
        dip_ten2 = Dip_tensor(coord3,coord4)
        a_s2 = get_sym_as(dip_ten2)
    
        Apref =0.0
        for i in range(5):
            Apref+=a_s1[i]*np.conjugate(a_s2[i])
    
    return hbar**2 * 1e-14*gamma**4 * SpecFunc(w,tc)*Apref/(r1**3 * r2**3) #The 1e-14 prefactor is due to the squared magnetic permeability (divided by 4*np.pi)


def K2_MatRep(freqs,tc,coords,Nspins,gamma,basis):
    """
    Function to compute the matrix representation of the K2+K-2 Linbladian contributions to the equations of motion
    """
    def MatRep(w,An,Am):
        return MatRepLib(w,An,Am,n_qubits=Nspins)
    
    #Construction of the matrix representation of linbladians...
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):

                    damp_rate = GammaRates(freqs[k]+freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    
                    dum=MatRep(basis,S_plus(i)*S_plus(j),S_plus(k)*S_plus(l))+MatRep(basis,S_minus(k)*S_minus(l),S_minus(i)*S_minus(j))
                    dum+=np.conjugate(np.transpose(dum))

                    Rel_Mat+=damp_rate*dum          
                    
    return Rel_Mat

def K1_MatRep(freqs,tc,coords,Nspins,gamma,basis):
    """
    Function that computes the matrix representation of relaxation for the K1+k-1 contributions to the equations of motion
    """
    def MatRep(w,An,Am):
        return MatRepLib(w,An,Am,n_qubits=Nspins)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    
                    damp_rate_l = GammaRates(freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_k = GammaRates(freqs[k],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    dum_l = MatRep(basis,Sz(i)*S_plus(j),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),Sz(i)*S_minus(j))
                    dum_l+= MatRep(basis,S_plus(i)*Sz(j),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),S_minus(i)*Sz(j))
                    dum_l+=np.conjugate(np.transpose(dum_l))

                    dum_k = MatRep(basis,S_plus(i)*Sz(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),S_minus(i)*Sz(j))
                    dum_k += MatRep(basis,Sz(i)*S_plus(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),Sz(i)*S_minus(j))
                    dum_k+=np.conjugate(np.transpose(dum_k))

                    Rel_Mat += damp_rate_l*dum_l+damp_rate_k*dum_k
                    
    return Rel_Mat

def K0_MatRep(freqs,tc,coords,Nspins,gamma,basis):
    """
    Function that computes the matrix representation of relaxation for the K0 contributions to the equations of motion
    """
    def MatRep(w,An,Am):
        return MatRepLib(w,An,Am,n_qubits=Nspins)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    damp_rate_0 = GammaRates(0,tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    dum_0 = -(8.0/3.0)*MatRep(basis,Sz(i)*Sz(j),Sz(k)*Sz(l))
                    dum_0 += (2.0/3.0)*(MatRep(basis,S_plus(i)*S_minus(j),Sz(k)*Sz(l))+MatRep(basis,Sz(k)*Sz(l),S_minus(i)*S_plus(j)))
                    dum_0 += np.conjugate(np.transpose(dum_0))

                    dum_diff = (2.0/3.0)*(MatRep(basis,Sz(i)*Sz(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),Sz(i)*Sz(j)))
                    dum_diff+= -(1.0/6.0)*(MatRep(basis,S_plus(i)*S_minus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_minus(i)*S_plus(j)))
                    dum_diff+=-(1.0/6.0)*(MatRep(basis,S_minus(i)*S_plus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_plus(i)*S_minus(j)))
                    dum_diff+=np.conjugate(np.transpose(dum_diff))

                    Rel_Mat+=damp_rate_0*dum_0+damp_rate_diff*dum_diff

    return -Rel_Mat

def Get_K2RatesAndOps(freqs,tc,coords,Nspins,gamma):

    List_Jump_Ops = []
    List_rates = []

    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):

                    damp_rate = GammaRates(freqs[k]+freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    List_Jump_Ops.append([S_plus(i)*S_plus(j),S_minus(k)*S_minus(l)])
                    List_rates.append(damp_rate)

                    List_Jump_Ops.append([S_minus(k)*S_minus(l),S_plus(i)*S_plus(j)])
                    List_rates.append(damp_rate)

                    List_Jump_Ops.append([S_plus(k)*S_plus(l),S_minus(i)*S_minus(j)])
                    List_rates.append(damp_rate)

                    List_Jump_Ops.append([S_minus(i)*S_minus(j),S_plus(k)*S_plus(l)])
                    List_rates.append(damp_rate)


    
    return List_rates, List_Jump_Ops

def Get_K1RatesAndOps(freqs,tc,coords,Nspins,gamma):
    """
    Function that computes the matrix representation of relaxation for the K1+k-1 contributions to the equations of motion
    """
    List_Jump_Ops = []
    List_rates = []

    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    
                    damp_rate_l = GammaRates(freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_k = GammaRates(freqs[k],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    List_Jump_Ops.append([Sz(i)*S_plus(j),Sz(k)*S_minus(l)])
                    List_rates.append(damp_rate_l)

                    List_Jump_Ops.append([Sz(k)*S_plus(l),Sz(i)*S_minus(j)])
                    List_rates.append(damp_rate_l)

                    List_Jump_Ops.append([Sz(k)*S_minus(l),Sz(i)*S_plus(j)])
                    List_rates.append(damp_rate_l)

                    List_Jump_Ops.append([Sz(i)*S_minus(j),Sz(k)*S_plus(l)])
                    List_rates.append(damp_rate_l)

                    List_Jump_Ops.append([S_plus(i)*Sz(j),Sz(k)*S_minus(l)])
                    List_rates.append(damp_rate_l)

                    List_Jump_Ops.append([Sz(k)*S_plus(l),S_minus(i)*Sz(j)])
                    List_rates.append(damp_rate_l)

                    List_Jump_Ops.append([Sz(k)*S_minus(l),S_plus(i)*Sz(j)])
                    List_rates.append(damp_rate_l)

                    List_Jump_Ops.append([S_minus(i)*Sz(j),Sz(k)*S_plus(l)])
                    List_rates.append(damp_rate_l)


                    List_Jump_Ops.append([S_plus(i)*Sz(j),S_minus(k)*Sz(l)])
                    List_rates.append(damp_rate_k)

                    List_Jump_Ops.append([S_plus(k)*Sz(l),S_minus(i)*Sz(j)])
                    List_rates.append(damp_rate_k)

                    List_Jump_Ops.append([S_minus(k)*Sz(l),S_plus(i)*Sz(j)])
                    List_rates.append(damp_rate_k)

                    List_Jump_Ops.append([S_minus(i)*Sz(j),S_plus(k)*Sz(l)])
                    List_rates.append(damp_rate_k)

                    List_Jump_Ops.append([Sz(i)*S_plus(j),S_minus(k)*Sz(l)])
                    List_rates.append(damp_rate_k)

                    List_Jump_Ops.append([S_plus(k)*Sz(l),Sz(i)*S_minus(j)])
                    List_rates.append(damp_rate_k)

                    List_Jump_Ops.append([S_minus(k)*Sz(l),Sz(i)*S_plus(j)])
                    List_rates.append(damp_rate_k)
                    
                    List_Jump_Ops.append([Sz(i)*S_minus(j),S_plus(k)*Sz(l)])
                    List_rates.append(damp_rate_k)

                    #dum_k = MatRep(basis,S_plus(i)*Sz(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),S_minus(i)*Sz(j))
                    #dum_k += MatRep(basis,Sz(i)*S_plus(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),Sz(i)*S_minus(j))
                    #dum_k+=np.conjugate(np.transpose(dum_k))

                    #Rel_Mat += damp_rate_l*dum_l+damp_rate_k*dum_k
                    
    return List_rates, List_Jump_Ops

def Get_K0RatesAndOps(freqs,tc,coords,Nspins,gamma):
    """
    Function that computes the matrix representation of relaxation for the K0 contributions to the equations of motion
    """

    List_Jump_Ops = []
    List_rates = []
    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):

                    damp_rate_0 = GammaRates(0,tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    List_Jump_Ops.append([Sz(i)*Sz(j),Sz(k)*Sz(l)])
                    List_rates.append((8.0/3.0)*damp_rate_0)
                    List_Jump_Ops.append([Sz(k)*Sz(l),Sz(i)*Sz(j)])
                    List_rates.append((8.0/3.0)*damp_rate_0)

                    List_Jump_Ops.append([S_plus(i)*S_minus(j),Sz(k)*Sz(l)])
                    List_rates.append(-(2.0/3.0)*damp_rate_0)
                    List_Jump_Ops.append([Sz(k)*Sz(l),S_minus(i)*S_plus(j)])
                    List_rates.append(-(2.0/3.0)*damp_rate_0)

                    List_Jump_Ops.append([Sz(k)*Sz(l),S_plus(i)*S_minus(j)])
                    List_rates.append(-(2.0/3.0)*damp_rate_0)
                    List_Jump_Ops.append([S_minus(i)*S_plus(j),Sz(k)*Sz(l)])
                    List_rates.append(-(2.0/3.0)*damp_rate_0)

                    List_Jump_Ops.append([Sz(i)*Sz(j),S_minus(k)*S_plus(l)])
                    List_rates.append(-(2.0/3.0)*damp_rate_diff)
                    List_Jump_Ops.append([S_plus(k)*S_minus(l),Sz(i)*Sz(j)])
                    List_rates.append(-(2.0/3.0)*damp_rate_diff)

                    List_Jump_Ops.append([S_minus(k)*S_plus(l),Sz(i)*Sz(j)])
                    List_rates.append(-(2.0/3.0)*damp_rate_diff)
                    List_Jump_Ops.append([Sz(i)*Sz(j),S_plus(k)*S_minus(l)])
                    List_rates.append(-(2.0/3.0)*damp_rate_diff)

                    List_Jump_Ops.append([S_plus(i)*S_minus(j),S_minus(k)*S_plus(l)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)
                    List_Jump_Ops.append([S_plus(k)*S_minus(l),S_minus(i)*S_plus(j)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)

                    List_Jump_Ops.append([S_minus(k)*S_plus(l),S_plus(i)*S_minus(j)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)
                    List_Jump_Ops.append([S_minus(i)*S_plus(j),S_plus(k)*S_minus(l)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)

                    List_Jump_Ops.append([S_minus(i)*S_plus(j),S_minus(k)*S_plus(l)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)
                    List_Jump_Ops.append([S_plus(k)*S_minus(l),S_plus(i)*S_minus(j)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)

                    List_Jump_Ops.append([S_minus(k)*S_plus(l),S_minus(i)*S_plus(j)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)
                    List_Jump_Ops.append([S_plus(i)*S_minus(j),S_plus(k)*S_minus(l)])
                    List_rates.append((1.0/6.0)*damp_rate_diff)


                    #dum_0 = -(8.0/3.0)*MatRep(basis,Sz(i)*Sz(j),Sz(k)*Sz(l))
                    #dum_0 += (2.0/3.0)*(MatRep(basis,S_plus(i)*S_minus(j),Sz(k)*Sz(l))+MatRep(basis,Sz(k)*Sz(l),S_minus(i)*S_plus(j)))
                    #dum_0 += np.conjugate(np.transpose(dum_0))

                    #dum_diff = (2.0/3.0)*(MatRep(basis,Sz(i)*Sz(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),Sz(i)*Sz(j)))
                    #dum_diff+= -(1.0/6.0)*(MatRep(basis,S_plus(i)*S_minus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_minus(i)*S_plus(j)))
                    #dum_diff+=-(1.0/6.0)*(MatRep(basis,S_minus(i)*S_plus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_plus(i)*S_minus(j)))
                    #dum_diff+=np.conjugate(np.transpose(dum_diff))

                    #Rel_Mat+=damp_rate_0*dum_0+damp_rate_diff*dum_diff

    return List_rates,List_Jump_Ops



def GetRelManySpins(freqs,coords,tc,gamma,basis):
    """
    Returns: a matrix representation of the Relaxation super-operator for many spins
    Args:
    freqs: Zeeman frequencies in Hz for all spins
    coords: array whose ith element corresponds to the coordinates of the ith spin
    tc: rotational correlation time in seconds
    gamma: the gyromagnetic constant of the the spins (assumed to be a homonuclear case) 
    basis: the orthonormalized basis to build the matrix representation
    """

    #computing the different contributions to the equation of motion...
    Nspins = len(freqs)

    K2 = K2_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("K2 type contributions finished")
    K1 = K1_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("K1 type contributions finished")
    K0 = K0_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("K0 type contributions finished")

    return 0.25*K2,0.25*K1,0.25*K0,0.25*(K2+K1+K0)                    

###Parallel versions of functions for calculation of the relaxation matrix...
"""
def MatRepLibParallel(basis, An, Am, n_qubits=2, num_workers=None):
    
    Nbasis = len(basis)
    
    # Helper function to compute a block of the matrix
    def compute_element(args):
        i, j = args
        return InnProd(basis[i], Linb_Channel(An, Am, basis[j]), n_qubits=n_qubits)

    # Create an empty matrix to hold the result
    MatRep = np.zeros((Nbasis, Nbasis), dtype=complex)
    
    # Generate the index pairs to parallelize over
    index_pairs = [(i, j) for i in range(Nbasis) for j in range(Nbasis)]
    
    # Use ProcessPoolExecutor to parallelize the computation of matrix elements
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each matrix element
        results = executor.map(compute_element, index_pairs)
    
    # Collect the results into the matrix
    for (i, j), result in zip(index_pairs, results):
        MatRep[i, j] = result

    return MatRep
"""
def compute_element(args):
    i, j, basis, An, Am, n_qubits = args
    return i, j, InnProd(basis[i], Linb_Channel(An, Am, basis[j]), n_qubits=n_qubits)

def MatRepLibParallel(basis, An, Am, n_qubits=2, num_workers=None):
    """
    Function to compute the matrix representation of a Linbladian contribution
    with parallelized computation.
    """
    Nbasis = len(basis)

    # Create an empty matrix to hold the result
    MatRep = np.zeros((Nbasis, Nbasis), dtype=complex)
    
    # Generate the index pairs to parallelize over
    index_pairs = [(i, j, basis, An, Am, n_qubits) for i in range(Nbasis) for j in range(Nbasis)]
    
    # Use ProcessPoolExecutor to parallelize the computation of matrix elements
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each matrix element and collect the results
        results = executor.map(compute_element, index_pairs)
    
    # Collect the results into the matrix
    for i, j, result in results:
        MatRep[i, j] = result

    return MatRep



def K2_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=None):
    
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)
    
    #Construction of the matrix representation of linbladians...
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):

                    damp_rate = GammaRates(freqs[k]+freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    
                    dum=MatRep(basis,S_plus(i)*S_plus(j),S_plus(k)*S_plus(l))+MatRep(basis,S_minus(k)*S_minus(l),S_minus(i)*S_minus(j))
                    dum+=np.conjugate(np.transpose(dum))

                    Rel_Mat+=damp_rate*dum          
                    
    return Rel_Mat


def K1_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=None):
    """
    Function that computes the matrix representation of relaxation for the K1+k-1 contributions to the equations of motion
    """
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    
                    damp_rate_l = GammaRates(freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_k = GammaRates(freqs[k],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    dum_l = MatRep(basis,Sz(i)*S_plus(j),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),Sz(i)*S_minus(j))
                    dum_l+= MatRep(basis,S_plus(i)*Sz(j),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),S_minus(i)*Sz(j))
                    dum_l+=np.conjugate(np.transpose(dum_l))

                    dum_k = MatRep(basis,S_plus(i)*Sz(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),S_minus(i)*Sz(j))
                    dum_k += MatRep(basis,Sz(i)*S_plus(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),Sz(i)*S_minus(j))
                    dum_k+=np.conjugate(np.transpose(dum_k))

                    Rel_Mat += damp_rate_l*dum_l+damp_rate_k*dum_k
                    
    return Rel_Mat


def K0_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=None):
    """
    Function that computes the matrix representation of relaxation for the K0 contributions to the equations of motion
    """
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    damp_rate_0 = GammaRates(0,tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    dum_0 = -(8.0/3.0)*MatRep(basis,Sz(i)*Sz(j),Sz(k)*Sz(l))
                    dum_0 += (2.0/3.0)*(MatRep(basis,S_plus(i)*S_minus(j),Sz(k)*Sz(l))+MatRep(basis,Sz(k)*Sz(l),S_minus(i)*S_plus(j)))
                    dum_0 += np.conjugate(np.transpose(dum_0))

                    dum_diff = (2.0/3.0)*(MatRep(basis,Sz(i)*Sz(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),Sz(i)*Sz(j)))
                    dum_diff+= -(1.0/6.0)*(MatRep(basis,S_plus(i)*S_minus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_minus(i)*S_plus(j)))
                    dum_diff+=-(1.0/6.0)*(MatRep(basis,S_minus(i)*S_plus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_plus(i)*S_minus(j)))
                    dum_diff+=np.conjugate(np.transpose(dum_diff))

                    Rel_Mat+=damp_rate_0*dum_0+damp_rate_diff*dum_diff

    return -Rel_Mat


def GetRelManySpinsParallel(freqs,coords,tc,gamma,basis,num_workers=None):
    """
    Returns: a matrix representation of the Relaxation super-operator for many spins
    Args:
    freqs: Zeeman frequencies in Hz for all spins
    coords: array whose ith element corresponds to the coordinates of the ith spin
    tc: rotational correlation time in seconds
    gamma: the gyromagnetic constant of the the spins (assumed to be a homonuclear case) 
    basis: the orthonormalized basis to build the matrix representation
    """

    #computing the different contributions to the equation of motion...
    Nspins = len(freqs)

    K2 = K2_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=num_workers)
    print("K2 type contributions finished, saving it...")
    Dict={'mat': K2}

    with open('Spin4_K2.pk', 'wb') as handle:
        pickle.dump(Dict, handle)

    K1 = K1_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=num_workers)
    print("K1 type contributions finished")
    Dict={'mat': K1}

    with open('Spin4_K1.pk','wb') as handle:
        pickle.dump(Dict,handle)


    K0 = K0_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=num_workers)
    print("K0 type contributions finished")
    Dict={'mat': K0}

    with open('Spin4_K0.pk','wb') as handle:
        pickle.dump(Dict,handle)

    return 0.25*(K2+K1+K0) #Notice that we multiply by a factor of 4!!
    #return K2 


