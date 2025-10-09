import numpy as np
import sys
sys.path.append('./')
import pickle

from basis_utils import MatRepLib, InnProd, Linb_Channel
from basis_utils import Sz,S_plus,S_minus
from openfermion import hermitian_conjugated

from concurrent.futures import ProcessPoolExecutor



def convert_operator_string(operator_str):
    # Mapping symbols to functions
    operators_map = {'+': S_plus, '-': S_minus, 'z': Sz}
    output = []

    # Parse the input string in pairs of [index][operator]
    i = 0
    while i < len(operator_str):
        if operator_str[i] == 'S':  # Check for the 'S' character
            index = int(operator_str[i+1])  # Extract index (digit after 'S')
            op_symbol = operator_str[i+2]   # Extract operator symbol
            func = operators_map[op_symbol]  # Find the corresponding function
            
            # Append the result of the function call as a string
            output.append(func(index))
            i += 3  # Move to the next operator pair
        else:
            i += 1
    op = 1

    for i in range(len(output)):
        op=output[i]*op
    # Join all function calls with a multiplication symbol
    return op#' * '.join(output)




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
    Note that gamma is a 1D array of 4 elemnts that contain the gyromagnetic ratios
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
    
    return hbar**2 * 1e-14*np.prod(gamma) * SpecFunc(w,tc)*Apref/(r1**3 * r2**3) #The 1e-14 prefactor is due to the squared magnetic permeability (divided by 4*np.pi)


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



def Get_K2RatesAndOps(freqs,tc,coords,Nspins,gamma,get_strs=True):

    List_Jump_Ops = []
    List_rates = []

    if get_strs:
        Str_Ops = []


    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):

                    damp_rate = GammaRates(freqs[k]+freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma[[i,j,k,l]])

                    List_Jump_Ops.append([S_plus(i)*S_plus(j),S_minus(k)*S_minus(l)])
                    List_rates.append(damp_rate)

                    List_Jump_Ops.append([S_minus(k)*S_minus(l),S_plus(i)*S_plus(j)])
                    List_rates.append(damp_rate)

                    List_Jump_Ops.append([S_plus(k)*S_plus(l),S_minus(i)*S_minus(j)])
                    List_rates.append(damp_rate)

                    List_Jump_Ops.append([S_minus(i)*S_minus(j),S_plus(k)*S_plus(l)])
                    List_rates.append(damp_rate)

                    if get_strs:
                        Str_Ops.append(['S+'+str(i)+' S+'+str(j),'S-'+str(k)+' S-'+str(l)])
                        Str_Ops.append(['S-'+str(k)+' S-'+str(l),'S+'+str(i)+' S+'+str(j)])
                        Str_Ops.append(['S+'+str(k)+' S+'+str(l),'S-'+str(i)+' S-'+str(j)])
                        Str_Ops.append(['S-'+str(i)+' S-'+str(j),'S+'+str(k)+' S+'+str(l)])


    if get_strs:
        return List_rates, List_Jump_Ops, Str_Ops
    else:
        return List_rates, List_Jump_Ops

def Get_K1RatesAndOps(freqs,tc,coords,Nspins,gamma,get_strs=True):
    """
    Function that computes the matrix representation of relaxation for the K1+k-1 contributions to the equations of motion
    """
    List_Jump_Ops = []
    List_rates = []

    if get_strs:
        Str_Ops = []

    
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    
                    damp_rate_l = GammaRates(freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma[[i,j,k,l]])
                    damp_rate_k = GammaRates(freqs[k],tc,coords[i],coords[j],coords[k],coords[l],gamma[[i,j,k,l]])

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

                    if get_strs:
                        Str_Ops.append(['Sz'+str(i)+' S+'+str(j), 'Sz'+str(k)+' S-'+str(l)])
                        Str_Ops.append(['Sz'+str(k)+' S+'+str(l), 'Sz'+str(i)+' S-'+str(j)])
                        Str_Ops.append(['Sz'+str(k)+' S-'+str(l), 'Sz'+str(i)+' S+'+str(j)])
                        Str_Ops.append(['Sz'+str(i)+' S-'+str(j), 'Sz'+str(k)+' S+'+str(l)])
                        Str_Ops.append(['S+'+str(i)+' Sz'+str(j), 'Sz'+str(k)+' S-'+str(l)])
                        Str_Ops.append(['Sz'+str(k)+' S+'+str(l), 'S-'+str(i)+' Sz'+str(j)])
                        Str_Ops.append(['Sz'+str(k)+' S-'+str(l), 'S+'+str(i)+' Sz'+str(j)])
                        Str_Ops.append(['S-'+str(i)+' Sz'+str(j), 'Sz'+str(k)+' S+'+str(l)])
                        Str_Ops.append(['S+'+str(i)+' Sz'+str(j), 'S-'+str(k)+' Sz'+str(l)])
                        Str_Ops.append(['S+'+str(k)+' Sz'+str(l), 'S-'+str(i)+' Sz'+str(j)])
                        Str_Ops.append(['S-'+str(k)+' Sz'+str(l), 'S+'+str(i)+' Sz'+str(j)])
                        Str_Ops.append(['S-'+str(i)+' Sz'+str(j), 'S+'+str(k)+' Sz'+str(l)])
                        Str_Ops.append(['Sz'+str(i)+' S+'+str(j), 'S-'+str(k)+' Sz'+str(l)])
                        Str_Ops.append(['S+'+str(k)+' Sz'+str(l), 'Sz'+str(i)+' S-'+str(j)])
                        Str_Ops.append(['S-'+str(k)+' Sz'+str(l), 'Sz'+str(i)+' S+'+str(j)])
                        Str_Ops.append(['Sz'+str(i)+' S-'+str(j), 'S+'+str(k)+' Sz'+str(l)])

    if get_strs:
        return List_rates, List_Jump_Ops, Str_Ops
    else:        
        return List_rates, List_Jump_Ops

def Get_K0RatesAndOps(freqs,tc,coords,Nspins,gamma,get_strs=True):
    """
    Function that computes the matrix representation of relaxation for the K0 contributions to the equations of motion
    """

    List_Jump_Ops = []
    List_rates = []
    
    if get_strs:
        Str_Ops = []


    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):

                    damp_rate_0 = GammaRates(0,tc,coords[i],coords[j],coords[k],coords[l],gamma[[i,j,k,l]])
                    damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma[[i,j,k,l]])

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

                    if get_strs:
                        Str_Ops.append(['Sz'+str(i)+' Sz'+str(j), 'Sz'+str(k)+' Sz'+str(l)])
                        Str_Ops.append(['Sz'+str(k)+' Sz'+str(l), 'Sz'+str(i)+' Sz'+str(j)])
                        Str_Ops.append(['S+'+str(i)+' S-'+str(j), 'Sz'+str(k)+' Sz'+str(l)])
                        Str_Ops.append(['Sz'+str(k)+' Sz'+str(l), 'S-'+str(i)+' S+'+str(j)])
                        Str_Ops.append(['Sz'+str(k)+' Sz'+str(l), 'S+'+str(i)+' S-'+str(j)])
                        Str_Ops.append(['S-'+str(i)+' S+'+str(j), 'Sz'+str(k)+' Sz'+str(l)])
                        Str_Ops.append(['Sz'+str(i)+' Sz'+str(j), 'S-'+str(k)+' S+'+str(l)])
                        Str_Ops.append(['S+'+str(k)+' S-'+str(l), 'Sz'+str(i)+' Sz'+str(j)])
                        Str_Ops.append(['S-'+str(k)+' S+'+str(l), 'Sz'+str(i)+' Sz'+str(j)])
                        Str_Ops.append(['Sz'+str(i)+' Sz'+str(j), 'S+'+str(k)+' S-'+str(l)])
                        Str_Ops.append(['S+'+str(i)+' S-'+str(j), 'S-'+str(k)+' S+'+str(l)])
                        Str_Ops.append(['S+'+str(k)+' S-'+str(l), 'S-'+str(i)+' S+'+str(j)])
                        Str_Ops.append(['S-'+str(k)+' S+'+str(l), 'S+'+str(i)+' S-'+str(j)])
                        Str_Ops.append(['S-'+str(i)+' S+'+str(j), 'S+'+str(k)+' S-'+str(l)])
                        Str_Ops.append(['S-'+str(i)+' S+'+str(j), 'S-'+str(k)+' S+'+str(l)])
                        Str_Ops.append(['S+'+str(k)+' S-'+str(l), 'S+'+str(i)+' S-'+str(j)])
                        Str_Ops.append(['S-'+str(k)+' S+'+str(l), 'S-'+str(i)+' S+'+str(j)])
                        Str_Ops.append(['S+'+str(i)+' S-'+str(j), 'S+'+str(k)+' S-'+str(l)])

    if get_strs:
        return List_rates, List_Jump_Ops, Str_Ops
    else:        
        return List_rates, List_Jump_Ops






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


def Loc_K2_MatRep(freqs,tc,coords,Nspins,gamma,basis):
    """
    Function to compute the matrix representation of the K2+K-2 Linbladian contributions to the equations of motion
    """
    def MatRep(w,An,Am):
        return MatRepLib(w,An,Am,n_qubits=Nspins)
    
    #Construction of the matrix representation of linbladians...
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    #for i in range(Nspins):
    #    for j in range(i+1,Nspins):
            
    for k in range(Nspins):
        for l in range(k+1,Nspins):

            damp_rate = GammaRates(freqs[k]+freqs[l],tc,coords[k],coords[l],coords[k],coords[l],gamma)
            
            dum=MatRep(basis,S_plus(k)*S_plus(l),S_plus(k)*S_plus(l))+MatRep(basis,S_minus(k)*S_minus(l),S_minus(k)*S_minus(l))
            dum+=np.conjugate(np.transpose(dum))

            Rel_Mat+=damp_rate*dum          
                    
    return 0.25*Rel_Mat


def Loc_K1_MatRep(freqs,tc,coords,Nspins,gamma,basis):
    """
    Compute the self-relxation channels coming from the K=1 type terms, for the 2 spin system, we obtain a Kite-type relxation matrix
    """
    def MatRep(w,An,Am):
        return MatRepLib(w,An,Am,n_qubits=Nspins)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)

    #for i in range(Nspins):
    #    for j in range(i+1,Nspins):
            
    for k in range(Nspins):
        for l in range(k+1,Nspins):
            damp_rate_l = GammaRates(freqs[l],tc,coords[k],coords[l],coords[k],coords[l],gamma)
            damp_rate_k = GammaRates(freqs[k],tc,coords[k],coords[l],coords[k],coords[l],gamma)

            dum_l = MatRep(basis,Sz(k)*S_plus(l),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),Sz(k)*S_minus(l))
            #dum_l+= MatRep(basis,S_plus(i)*Sz(j),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),S_minus(i)*Sz(j))
            dum_l+=np.conjugate(np.transpose(dum_l))

            dum_k = MatRep(basis,S_plus(k)*Sz(l),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),S_minus(k)*Sz(l))
            #dum_k += MatRep(basis,Sz(i)*S_plus(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),Sz(i)*S_minus(j))
            dum_k+=np.conjugate(np.transpose(dum_k))

            Rel_Mat += damp_rate_l*dum_l+damp_rate_k*dum_k

    return 0.25*Rel_Mat

def Loc_K0_MatRep(freqs,tc,coords,Nspins,gamma,basis):
     
    def MatRep(w,An,Am):
        return MatRepLib(w,An,Am,n_qubits=Nspins)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)

    #for i in range(Nspins):
    #    for j in range(i+1,Nspins):
            
    for k in range(Nspins):
        for l in range(k+1,Nspins):
            damp_rate_0 = GammaRates(0,tc,coords[k],coords[l],coords[k],coords[l],gamma)
            damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[k],coords[l],coords[k],coords[l],gamma)

            dum_0 = -(8.0/3.0)*MatRep(basis,Sz(k)*Sz(l),Sz(k)*Sz(l))
            #dum_0 += (2.0/3.0)*(MatRep(basis,S_plus(i)*S_minus(j),Sz(k)*Sz(l))+MatRep(basis,Sz(k)*Sz(l),S_minus(i)*S_plus(j))) adding this removes the kite structure
            
            dum_0 += np.conjugate(np.transpose(dum_0))

            dum_diff= -(1.0/6.0)*(MatRep(basis,S_plus(k)*S_minus(l),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_minus(k)*S_plus(l)))
            #dum_diff+=-(1.0/6.0)*(MatRep(basis,S_minus(i)*S_plus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_plus(i)*S_minus(j))) #adding this removes the kite structure
            dum_diff+=np.conjugate(np.transpose(dum_diff))


            Rel_Mat+=damp_rate_0*dum_0+damp_rate_diff*dum_diff

    return -0.25*Rel_Mat


def KiteRelMatrixMany(freqs,tc,coords,Nspins,gamma,basis):

    K2 = Loc_K2_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("Finished computing the K2 type contributions")
    K1 = Loc_K1_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("Finished computing the K1 type contributions")
    K0 = Loc_K0_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("Finished computing the K0 type contributions")
    

    return K2+K1+K0

def ApproxKiteMany(freqs,tc,coords,Nspins,gamma,basis):

    K1 = Loc_K1_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("Finished computing the K1 type contributions")

    K0 = Loc_K0_MatRep(freqs,tc,coords,Nspins,gamma,basis)
    print("Finished computing the K0 type contributions")
    

    return K1+K0


## Parallel version of previous functions...
def Loc_K2_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=None):
    """
    Function to compute the matrix representation of the K2+K-2 Linbladian contributions to the equations of motion
    """
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)
    
    #Construction of the matrix representation of linbladians...
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    
    #for i in range(Nspins):
    #    for j in range(i+1,Nspins):
            
    for k in range(Nspins):
        for l in range(k+1,Nspins):

            damp_rate = GammaRates(freqs[k]+freqs[l],tc,coords[k],coords[l],coords[k],coords[l],gamma)
            
            dum=MatRep(basis,S_plus(k)*S_plus(l),S_plus(k)*S_plus(l))+MatRep(basis,S_minus(k)*S_minus(l),S_minus(k)*S_minus(l))
            dum+=np.conjugate(np.transpose(dum))

            Rel_Mat+=damp_rate*dum          
                    
    return 0.25*Rel_Mat


def Loc_K1_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=None):
    """
    Compute the self-relxation channels coming from the K=1 type terms, for the 2 spin system, we obtain a Kite-type relxation matrix
    """
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)

    #for i in range(Nspins):
    #    for j in range(i+1,Nspins):
            
    for k in range(Nspins):
        for l in range(k+1,Nspins):
            damp_rate_l = GammaRates(freqs[l],tc,coords[k],coords[l],coords[k],coords[l],gamma)
            damp_rate_k = GammaRates(freqs[k],tc,coords[k],coords[l],coords[k],coords[l],gamma)

            dum_l = MatRep(basis,Sz(k)*S_plus(l),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),Sz(k)*S_minus(l))
            #dum_l+= MatRep(basis,S_plus(i)*Sz(j),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),S_minus(i)*Sz(j))
            dum_l+=np.conjugate(np.transpose(dum_l))

            dum_k = MatRep(basis,S_plus(k)*Sz(l),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),S_minus(k)*Sz(l))
            #dum_k += MatRep(basis,Sz(i)*S_plus(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),Sz(i)*S_minus(j))
            dum_k+=np.conjugate(np.transpose(dum_k))

            Rel_Mat += damp_rate_l*dum_l+damp_rate_k*dum_k

    return 0.25*Rel_Mat

def Loc_K0_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=None):
     
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)

    #for i in range(Nspins):
    #    for j in range(i+1,Nspins):
            
    for k in range(Nspins):
        for l in range(k+1,Nspins):
            damp_rate_0 = GammaRates(0,tc,coords[k],coords[l],coords[k],coords[l],gamma)
            damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[k],coords[l],coords[k],coords[l],gamma)

            dum_0 = -(8.0/3.0)*MatRep(basis,Sz(k)*Sz(l),Sz(k)*Sz(l))
            #dum_0 += (2.0/3.0)*(MatRep(basis,S_plus(i)*S_minus(j),Sz(k)*Sz(l))+MatRep(basis,Sz(k)*Sz(l),S_minus(i)*S_plus(j))) adding this removes the kite structure
            
            dum_0 += np.conjugate(np.transpose(dum_0))

            dum_diff= -(1.0/6.0)*(MatRep(basis,S_plus(k)*S_minus(l),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_minus(k)*S_plus(l)))
            #dum_diff+=-(1.0/6.0)*(MatRep(basis,S_minus(i)*S_plus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_plus(i)*S_minus(j))) #adding this removes the kite structure
            dum_diff+=np.conjugate(np.transpose(dum_diff))


            Rel_Mat+=damp_rate_0*dum_0+damp_rate_diff*dum_diff

    return -0.25*Rel_Mat


def KiteRelMatrixManyParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=None):

    K2 = Loc_K2_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis,num_workers=num_workers)
    print("Finished computing the K2 type contributions")
    K1 = Loc_K1_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis)
    print("Finished computing the K1 type contributions")
    K0 = Loc_K0_MatRepParallel(freqs,tc,coords,Nspins,gamma,basis)
    print("Finished computing the K0 type contributions")
    

    return K2+K1+K0



def get_chemical_shifts(gamma,B0,zeeman_scalars):
    """ 
    Returns: list of chemical shifts
    Args: 
    gamma: the gyromagnetic ratio of the nuclei, assumed to be the same for all
    B0: streng of the magnetic field
    zeeman_scalars: array that contain the isotropic chemical shifts, in ppm
    """
    w0 = -gamma*B0

    list_chem = []

    for i in range(len(zeeman_scalars)):
        list_chem.append(w0*zeeman_scalars[i]/1e6)

    return list_chem

def str_S_plus(i):
    return 'S'+str(i)+'+'

def str_S_minus(i):
    return 'S'+str(i)+'-'

def str_Sz(i):
    return 'S'+str(i)+'z'

#For latex support...
def lat_S_plus(i):
    return r'S_{'+str(i)+'+}'

def lat_S_minus(i):
    return r'S_{'+str(i)+'-}'

def lat_Sz(i):
    return r'S_{'+str(i)+'z}'

def J_plus(str_op1,str_op2):
    #print("Entering J_plus")
    string = 'J_{+}\left('+str_op1+','+str_op2+'\\right)'

    return string

def J_minus(str_op1,str_op2):
    string = 'J_{-}\left('+str_op1+','+str_op2+'\\right)'

    return string

def sym_omega(i):
    string = r'\omega_{'+str(i)+'}'
    return string

def SymGamma(i,j,k,l,freq):

    string = r'\Gamma^{('+str(i)+','+str(j)+','+str(k)+','+str(l)+')}'+'('+freq+')'
    return string

def one_fourth():
    return r'\frac{1}{4}'
def two_thirds():
    return r'\frac{2}{3}'
def one_sixth():
    return r'\frac{1}{6}'
def one_24():
    return r'\frac{1}{24}'


def Get_Det_And_Rates(freqs,tc,coords,Nspins,gamma,chemical_shifts):
    """
    Returns: 1) list of strings for pairs of jump operators that define a relaxation channel 2) its associated damping rate and 3) its oscillatory rate in the rotating frame of the the zeroth-order
    Hamiltonian
    Args:
    freqs: is the list of the complete isotropic Zeeman frequencies for the spins
    tc: correlation time for the classical rotational bath (in seconds)
    coords: list that contains the cartesian coordinates of the spins (in meters)
    Nspins: number of spins 
    gamma: gyromagnetic ratio for spins (assuming an homonuclear scenario)
    chemical shifts: list of chemical shifts for the spins
    """

    list_jumps = []
    list_damp_rates = []
    list_dets = []

    for i in range(Nspins):
        for j in range(i+1,Nspins):

            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    delt_i = chemical_shifts[i]
                    delt_j = chemical_shifts[j]
                    delt_k =  chemical_shifts[k]
                    delt_l = chemical_shifts[l]

                    damp_rate_sum = GammaRates(freqs[k]+freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_k = GammaRates(freqs[k],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_l = GammaRates(freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_0 = GammaRates(0,tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    list_jumps.append([str_S_plus(i)+str_S_plus(j),str_S_minus(k)+str_S_minus(l)])
                    list_jumps.append([str_S_minus(k)+str_S_minus(l),str_S_plus(i)+str_S_plus(j)])
                    list_jumps.append([str_Sz(i)+str_S_plus(j),str_Sz(k)+str_S_minus(l)])

                    list_jumps.append([str_Sz(k)+str_S_minus(l), str_Sz(i)+str_S_plus(j)])
                    list_jumps.append([str_S_plus(i)+str_Sz(j), str_Sz(k)+str_S_minus(l)])
                    list_jumps.append([str_Sz(k)+str_S_minus(l), str_S_plus(i)+str_Sz(j)])
                    list_jumps.append([str_S_plus(i)+str_Sz(j), str_S_minus(k)+str_Sz(l)])
                    list_jumps.append([str_S_minus(k)+str_Sz(l), str_S_plus(i)+str_Sz(j)])
                    list_jumps.append([str_Sz(i)+str_S_plus(j), str_S_minus(k)+str_Sz(l)])
                    list_jumps.append([str_S_minus(k)+str_Sz(l), str_Sz(i)+str_S_plus(j)])
                    list_jumps.append([str_Sz(i)+str_Sz(j), str_Sz(k)+str_Sz(l)])
                    list_jumps.append([str_S_plus(i)+str_S_minus(j), str_Sz(k)+str_Sz(l)])
                    list_jumps.append([str_Sz(k)+str_Sz(l), str_S_plus(i)+str_S_minus(j)])
                    list_jumps.append([str_Sz(i)+str_Sz(j), str_S_minus(k)+str_S_plus(l)])
                    list_jumps.append([str_S_minus(k)+str_S_plus(l), str_Sz(i)+str_Sz(j)])
                    list_jumps.append([str_S_plus(i)+str_S_minus(j), str_S_minus(k)+str_S_plus(l)])
                    list_jumps.append([str_S_minus(k)+str_S_plus(l), str_S_plus(i)+str_S_minus(j)])
                    list_jumps.append([str_S_minus(i)+str_S_plus(j), str_S_minus(k)+str_S_plus(l)])
                    list_jumps.append([str_S_minus(k)+str_S_plus(l), str_S_minus(i)+str_S_plus(j)])
                    ###list of damping rates...
                    list_damp_rates.append(damp_rate_sum)
                    list_damp_rates.append(damp_rate_sum)
                    list_damp_rates.append(damp_rate_l)
                    list_damp_rates.append(damp_rate_l)
                    list_damp_rates.append(damp_rate_l)
                    list_damp_rates.append(damp_rate_l)
                    list_damp_rates.append(damp_rate_k)
                    list_damp_rates.append(damp_rate_k)
                    list_damp_rates.append(damp_rate_k)
                    list_damp_rates.append(damp_rate_k)
                    list_damp_rates.append((8.0/3.0)*damp_rate_0)
                    list_damp_rates.append(-(2.0/3.0)*damp_rate_0) #TODO: verify signs...
                    list_damp_rates.append(-(2.0/3.0)*damp_rate_0)
                    list_damp_rates.append(-(2.0/3.0)*damp_rate_diff)
                    list_damp_rates.append(-(2.0/3.0)*damp_rate_diff)
                    list_damp_rates.append((1.0/6.0)*damp_rate_diff)
                    list_damp_rates.append((1.0/6.0)*damp_rate_diff)
                    list_damp_rates.append((1.0/6.0)*damp_rate_diff)
                    list_damp_rates.append((1.0/6.0)*damp_rate_diff)
                    #list of detunings...
                    list_dets.append(np.abs(delt_i+delt_j-delt_k-delt_l))
                    list_dets.append(np.abs(delt_i+delt_j-delt_k-delt_l))
                    list_dets.append(np.abs(delt_j-delt_l))
                    list_dets.append(np.abs(delt_j-delt_l))
                    list_dets.append(np.abs(delt_i-delt_l))
                    list_dets.append(np.abs(delt_i-delt_l))
                    list_dets.append(np.abs(delt_i-delt_k))
                    list_dets.append(np.abs(delt_i-delt_k))
                    list_dets.append(np.abs(delt_j-delt_k))
                    list_dets.append(np.abs(delt_j-delt_k))
                    list_dets.append(0)
                    list_dets.append(np.abs(delt_i-delt_j))
                    list_dets.append(np.abs(delt_i-delt_j))
                    list_dets.append(np.abs(delt_l-delt_k))
                    list_dets.append(np.abs(delt_l-delt_k))
                    list_dets.append(np.abs(delt_i+delt_l-delt_j-delt_k))
                    list_dets.append(np.abs(delt_i+delt_l-delt_j-delt_k))
                    list_dets.append(np.abs(delt_j+delt_l-delt_i-delt_k))
                    list_dets.append(np.abs(delt_j+delt_l-delt_i-delt_k))

    return list_jumps, list_damp_rates, list_dets

#####For the purposes of generating text in latex format that can be easily compiled, we adapt the previous fucntion accordingly...
def Get_Det_And_Rates_latex(freqs,tc,coords,Nspins,gamma,chemical_shifts):
    """
    Returns: 1) list of strings for pairs of jump operators that define a relaxation channel 2) its associated damping rate and 3) its oscillatory rate in the rotating frame of the the zeroth-order
    Hamiltonian
    Args:
    freqs: is the list of the complete isotropic Zeeman frequencies for the spins
    tc: correlation time for the classical rotational bath (in seconds)
    coords: list that contains the cartesian coordinates of the spins (in meters)
    Nspins: number of spins 
    gamma: gyromagnetic ratio for spins (assuming an homonuclear scenario)
    chemical shifts: list of chemical shifts for the spins
    """

    list_jumps = []
    list_damp_rates = []
    list_symb_rates = []
    list_dets = []

    for i in range(Nspins):
        for j in range(i+1,Nspins):

            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    delt_i = chemical_shifts[i]
                    delt_j = chemical_shifts[j]
                    delt_k =  chemical_shifts[k]
                    delt_l = chemical_shifts[l]

                    damp_rate_sum = GammaRates(freqs[k]+freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_k = GammaRates(freqs[k],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_l = GammaRates(freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_0 = GammaRates(0,tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    #list_jumps.append([str_S_plus(i)+str_S_plus(j),str_S_minus(k)+str_S_minus(l)])
                    #Grouping canonical linbladian channels
                    list_jumps.append(J_plus(lat_S_plus(i)+lat_S_plus(j),lat_S_plus(k)+lat_S_plus(l)) \
                                      +','+J_plus(lat_S_minus(k)+lat_S_minus(l),lat_S_minus(i)+lat_S_minus(j)))
                    
                    list_symb_rates.append(one_fourth()+SymGamma(i,j,k,l,sym_omega(k)+'+'+sym_omega(l)))
                    list_damp_rates.append(0.25*damp_rate_sum)
                    list_dets.append(np.abs(delt_i+delt_j-delt_k-delt_l))
                    ##
                    list_jumps.append(J_minus(lat_S_plus(i)+lat_S_plus(j),lat_S_plus(k)+lat_S_plus(l)) \
                                      +','+J_minus(lat_S_minus(k)+lat_S_minus(l),lat_S_minus(i)+lat_S_minus(j)))
                    
                    list_symb_rates.append('-'+one_fourth()+SymGamma(i,j,k,l,sym_omega(k)+'+'+sym_omega(l)))
                    list_damp_rates.append(-0.25*damp_rate_sum)
                    list_dets.append(np.abs(delt_i+delt_j-delt_k-delt_l))
                    ##
                    list_jumps.append(J_plus(lat_Sz(i)+lat_S_plus(j),lat_Sz(k)+lat_S_plus(l)) \
                                      +','+J_plus(lat_Sz(k)+lat_S_minus(l),lat_Sz(i)+lat_S_minus(j)))
                    
                    list_symb_rates.append(one_fourth()+SymGamma(i,j,k,l,sym_omega(l)))
                    list_damp_rates.append(0.25*damp_rate_l)
                    list_dets.append(np.abs(delt_j-delt_l))
                    ##
                    list_jumps.append(J_minus(lat_Sz(i)+lat_S_plus(j),lat_Sz(k)+lat_S_plus(l)) \
                                      +','+J_minus(lat_Sz(k)+lat_S_minus(l),lat_Sz(i)+lat_S_minus(j)))
                    list_symb_rates.append('-'+one_fourth()+SymGamma(i,j,k,l,sym_omega(l)))
                    list_damp_rates.append(-0.25*damp_rate_l)
                    list_dets.append(np.abs(delt_j-delt_l))
                    ##
                    list_jumps.append(J_plus(lat_S_plus(i)+lat_Sz(j),lat_Sz(k)+lat_S_plus(l)) \
                                      +','+J_plus(lat_Sz(k)+lat_S_minus(l),lat_S_minus(i)+lat_Sz(j)))
                    list_symb_rates.append(one_fourth()+SymGamma(i,j,k,l,sym_omega(l)))
                    list_damp_rates.append(0.25*damp_rate_l)
                    ##
                    list_jumps.append(J_minus(lat_S_plus(i)+lat_Sz(j),lat_Sz(k)+lat_S_plus(l)) \
                                      +','+J_minus(lat_Sz(k)+lat_S_minus(l),lat_S_minus(i)+lat_Sz(j)))
                    list_symb_rates.append('-'+one_fourth()+SymGamma(i,j,k,l,sym_omega(l)))
                    list_damp_rates.append(-0.25*damp_rate_l)
                    list_dets.append(np.abs(delt_i-delt_l))
                    ##Starting k terms..
                    list_jumps.append(J_plus(lat_S_plus(i)+lat_Sz(j),lat_S_plus(k)+lat_Sz(l)) \
                                      +','+J_plus(lat_S_minus(k)+lat_Sz(l),lat_S_minus(i)+lat_Sz(j)))
                    list_symb_rates.append(one_fourth()+SymGamma(i,j,k,l,sym_omega(k)))
                    list_damp_rates.append(0.25*damp_rate_k)
                    list_dets.append(np.abs(delt_i-delt_k))
                    ##
                    list_jumps.append(J_minus(lat_S_plus(i)+lat_Sz(j),lat_S_plus(k)+lat_Sz(l)) \
                                      +','+J_minus(lat_S_minus(k)+lat_Sz(l),lat_S_minus(i)+lat_Sz(j)))
                    list_symb_rates.append('-'+one_fourth()+SymGamma(i,j,k,l,sym_omega(k)))
                    list_damp_rates.append(-0.25*damp_rate_k)
                    list_dets.append(np.abs(delt_i-delt_k))
                    #second k terms..
                    list_jumps.append(J_plus(lat_Sz(i)+lat_S_plus(j),lat_S_plus(k)+lat_Sz(l)) \
                                      +','+J_plus(lat_S_minus(k)+lat_Sz(l),lat_Sz(i)+lat_S_plus(j)))
                    list_symb_rates.append(one_fourth()+SymGamma(i,j,k,l,sym_omega(k)))
                    list_damp_rates.append(0.25*damp_rate_k)
                    list_dets.append(np.abs(delt_j-delt_k))
                    ##
                    list_jumps.append(J_minus(lat_Sz(i)+lat_S_plus(j),lat_S_plus(k)+lat_Sz(l)) \
                                      +','+J_minus(lat_S_minus(k)+lat_Sz(l),lat_Sz(i)+lat_S_plus(j)))
                    list_symb_rates.append('-'+one_fourth()+SymGamma(i,j,k,l,sym_omega(k)))
                    list_damp_rates.append(-0.25*damp_rate_k)
                    list_dets.append(np.abs(delt_j-delt_k))

                    #The K0 terms...
                    list_jumps.append(J_plus(lat_Sz(i)+lat_Sz(j),lat_Sz(k)+lat_Sz(l)))
                    list_symb_rates.append(two_thirds()+SymGamma(i,j,k,l,'0'))
                    list_damp_rates.append((2.0/3.0)*damp_rate_0)
                    list_dets.append(0)
                    ##
                    list_jumps.append(J_minus(lat_Sz(i)+lat_Sz(j),lat_Sz(k)+lat_Sz(l)))
                    list_symb_rates.append('-'+two_thirds()+SymGamma(i,j,k,l,'0'))
                    list_damp_rates.append(-(2.0/3.0)*damp_rate_0)
                    list_dets.append(0)
                    ##
                    list_jumps.append(J_plus(lat_S_plus(i)+lat_S_minus(j),lat_Sz(k)+lat_Sz(l)) \
                                      +','+J_plus(lat_Sz(k)+lat_Sz(l),lat_S_minus(i)+lat_S_plus(j)))
                    list_symb_rates.append('-'+one_sixth()+SymGamma(i,j,k,l,'0'))
                    list_damp_rates.append(-(1.0/6.0)*damp_rate_0)
                    list_dets.append(0)
                    ##
                    list_jumps.append(J_minus(lat_S_plus(i)+lat_S_minus(j),lat_Sz(k)+lat_Sz(l)) \
                                      +','+J_minus(lat_Sz(k)+lat_Sz(l),lat_S_minus(i)+lat_S_plus(j)))
                    list_symb_rates.append(one_sixth()+SymGamma(i,j,k,l,'0'))
                    list_damp_rates.append((1.0/6.0)*damp_rate_0)
                    list_dets.append(delt_i-delt_j)
                    #first diff damp
                    list_jumps.append(J_plus(lat_Sz(i)+lat_Sz(j),lat_S_plus(k)+lat_S_minus(l)) \
                                      +','+J_plus(lat_S_minus(k)+lat_S_plus(l),lat_Sz(i)+lat_Sz(j)))
                    list_symb_rates.append('-'+one_sixth()+SymGamma(i,j,k,l,sym_omega(k)+'-'+sym_omega(l)))
                    list_damp_rates.append(-(1.0/6.0)*damp_rate_diff)
                    list_dets.append(delt_i-delt_j)
                    ##
                    list_jumps.append(J_minus(lat_Sz(i)+lat_Sz(j),lat_S_plus(k)+lat_S_minus(l)) \
                                      +','+J_minus(lat_S_minus(k)+lat_S_plus(l),lat_Sz(i)+lat_Sz(j)))
                    list_symb_rates.append(one_sixth()+SymGamma(i,j,k,l,sym_omega(k)+'-'+sym_omega(l)))
                    list_damp_rates.append((1.0/6.0)*damp_rate_diff)
                    list_dets.append(delt_k-delt_l)
                    #second diff damp
                    list_jumps.append(J_plus(lat_S_plus(i)+lat_S_plus(j),lat_S_plus(k)+lat_S_minus(l)) \
                                      +','+J_plus(lat_S_minus(k)+lat_S_plus(l),lat_S_minus(i)+lat_S_plus(j)))
                    list_symb_rates.append(one_24()+SymGamma(i,j,k,l,sym_omega(k)+'-'+sym_omega(l)))
                    list_damp_rates.append((1.0/24.0)*damp_rate_diff)
                    list_dets.append(delt_i+delt_k-delt_l-delt_j)
                    ##
                    list_jumps.append(J_minus(lat_S_plus(i)+lat_S_plus(j),lat_S_plus(k)+lat_S_minus(l)) \
                                      +','+J_minus(lat_S_minus(k)+lat_S_plus(l),lat_S_minus(i)+lat_S_plus(j)))
                    list_symb_rates.append('-'+one_24()+SymGamma(i,j,k,l,sym_omega(k)+'-'+sym_omega(l)))
                    list_damp_rates.append(-(1.0/24.0)*damp_rate_diff)
                    list_dets.append(delt_i+delt_k-delt_l-delt_j)
                    ##last terms...
                    list_jumps.append(J_plus(lat_S_minus(i)+lat_S_plus(j),lat_S_plus(k)+lat_S_minus(l)) \
                                      +','+J_plus(lat_S_minus(k)+lat_S_plus(l),lat_S_plus(i)+lat_S_minus(j)))

                    list_symb_rates.append(one_24()+SymGamma(i,j,k,l,sym_omega(k)+'-'+sym_omega(l)))
                    list_damp_rates.append((1.0/24.0)*damp_rate_diff)
                    list_dets.append(delt_j+delt_k-delt_l-delt_i)
                    ##
                    list_jumps.append(J_minus(lat_S_minus(i)+lat_S_plus(j),lat_S_plus(k)+lat_S_minus(l)) \
                                      +','+J_minus(lat_S_minus(k)+lat_S_plus(l),lat_S_plus(i)+lat_S_minus(j)))

                    list_symb_rates.append('-'+one_24()+SymGamma(i,j,k,l,sym_omega(k)+'-'+sym_omega(l)))
                    list_damp_rates.append(-(1.0/24.0)*damp_rate_diff)
                    list_dets.append(delt_j+delt_k-delt_l-delt_i)


    return list_jumps, list_symb_rates, list_damp_rates, list_dets





def RelMat_from_ops_and_rates(jump_ops,rates,basis,Nspins):
    """ 
    Returns the relaxation matrix out of a list of jump operators and its corresponding rates. 
    """
    if len(jump_ops)!=len(rates):
        print("Number of jump operators must match number of damping rates")
        exit()

    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    for i in range(len(rates)):
        JOp1 = convert_operator_string(jump_ops[i][0])
        JOp2 = hermitian_conjugated(convert_operator_string(jump_ops[i][1]))

        Rel_chan = rates[i]*MatRepLib(basis,JOp1,JOp2,n_qubits=Nspins)
        Rel_chan+= Rel_chan.conjugate().T

        Rel_Mat+=Rel_chan


    return Rel_Mat

###Parallel version of previous function....
def RelMat_from_ops_and_rates_parallel(jump_ops,rates,basis,Nspins,num_workers=None):
    """ 
    Returns the relaxation matrix out of a list of jump operators and its corresponding rates. 
    """
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)


    if len(jump_ops)!=len(rates):
        print("Number of jump operators must match number of damping rates")
        exit()

    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    for i in range(len(rates)):
        JOp1 = convert_operator_string(jump_ops[i][0])
        JOp2 = hermitian_conjugated(convert_operator_string(jump_ops[i][1]))

        Rel_chan = rates[i]*MatRep(basis,JOp1,JOp2)
        Rel_chan+= Rel_chan.conjugate().T

        Rel_Mat+=Rel_chan


    return Rel_Mat


#Fur the purposes of rapid computation and debugging...
def MatRepLib_Eff(basis,An,Am,non_van_idxs,n_qubits=2):
    """
     
    """

    Nbasis = len(basis)
    MatRep = np.zeros([Nbasis,Nbasis],dtype=complex)

    for idxs in non_van_idxs:
        i = idxs[0]
        j = idxs[1]
        
        MatRep[i,j] = InnProd(basis[i],Linb_Channel(An,Am,basis[j]),n_qubits=n_qubits)
    
    return MatRep


def RelMat_from_ops_and_rates_Eff(jump_ops,rates,basis,Nspins,non_van_idxs):
    """ 
    Returns the relaxation matrix out of a list of jump operators and its corresponding rates. 
    """
    if len(jump_ops)!=len(rates):
        print("Number of jump operators must match number of damping rates")
        exit()

    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)
    for i in range(len(rates)):
        JOp1 = convert_operator_string(jump_ops[i][0])
        JOp2 = hermitian_conjugated(convert_operator_string(jump_ops[i][1]))

        Rel_chan = rates[i]*MatRepLib_Eff(basis,JOp1,JOp2,non_van_idxs,n_qubits=Nspins)
        Rel_chan+= Rel_chan.conjugate().T

        Rel_Mat+=Rel_chan


    return Rel_Mat

#Parallel version of previous function...

def compute_rel_chan(rate, jump_op, basis, non_van_idxs, Nspins):
    JOp1 = convert_operator_string(jump_op[0])
    JOp2 = hermitian_conjugated(convert_operator_string(jump_op[1]))

    Rel_chan = rate * MatRepLib_Eff(basis, JOp1, JOp2, non_van_idxs, n_qubits=Nspins)
    Rel_chan += Rel_chan.conjugate().T
    return Rel_chan

def RelMat_from_ops_and_rates_Eff_parallel(jump_ops, rates, basis, Nspins, non_van_idxs, num_workers=None):
    """ 
    Returns the relaxation matrix from a list of jump operators and its corresponding rates.
    """
    if len(jump_ops) != len(rates):
        print("Number of jump operators must match number of damping rates")
        exit()

    Rel_Mat = np.zeros([len(basis), len(basis)], dtype=complex)

    # Parallelize the computation of each relaxation channel with a specified number of workers.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            compute_rel_chan,
            rates,
            jump_ops,
            [basis] * len(rates),
            [non_van_idxs] * len(rates),
            [Nspins] * len(rates)
        )
        
    # Sum up the results to form the relaxation matrix.
    for Rel_chan in results:
        Rel_Mat += Rel_chan

    return Rel_Mat





###       #DEPRECATED  ##############################################################################
def Kite_relMat(freqs,tc,coords,Nspins,gamma,basis):
    "In construction toi verify yhe Kite form of relaxation matrix"


    K2 = K2_MatRep(freqs,tc,coords,Nspins,gamma,basis)

    K1 = SelfRelaxK1MatRep(freqs,tc,coords,Nspins,gamma,basis)
    Zz_cont = ZZ_RelChanMatRep(freqs,tc,coords,Nspins,gamma,basis)

    return 0.25*K2+0.25*K1+Zz_cont



def ZZ_RelChanMatRep(freqs,tc,coords,Nspins,gamma,basis):
    """
    Compute the Linbladian relaxation channels with Jump operators that consist of products of Z operators only 
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
                    #dum_0 += (2.0/3.0)*(MatRep(basis,S_plus(i)*S_minus(j),Sz(k)*Sz(l))+MatRep(basis,Sz(k)*Sz(l),S_minus(i)*S_plus(j))) adding this removes the kite structure
                    
                    dum_0 += np.conjugate(np.transpose(dum_0))

                    dum_diff= -(1.0/6.0)*(MatRep(basis,S_plus(i)*S_minus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_minus(i)*S_plus(j)))
                    #dum_diff+=-(1.0/6.0)*(MatRep(basis,S_minus(i)*S_plus(j),S_plus(k)*S_minus(l))+MatRep(basis,S_minus(k)*S_plus(l),S_plus(i)*S_minus(j))) #adding this removes the kite structure
                    dum_diff+=np.conjugate(np.transpose(dum_diff))


                    Rel_Mat+=damp_rate_0*dum_0+damp_rate_diff*dum_diff

    return -0.25*Rel_Mat

def SelfRelaxK1MatRep(freqs,tc,coords,Nspins,gamma,basis):
    """
    Compute the self-relxation channels coming from the K=1 type terms, for the 2 spin system, we obtain a Kite-type relxation matrix
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
                    #dum_l+= MatRep(basis,S_plus(i)*Sz(j),Sz(k)*S_plus(l))+MatRep(basis,Sz(k)*S_minus(l),S_minus(i)*Sz(j))
                    dum_l+=np.conjugate(np.transpose(dum_l))

                    dum_k = MatRep(basis,S_plus(i)*Sz(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),S_minus(i)*Sz(j))
                    #dum_k += MatRep(basis,Sz(i)*S_plus(j),S_plus(k)*Sz(l))+MatRep(basis,S_minus(k)*Sz(l),Sz(i)*S_minus(j))
                    dum_k+=np.conjugate(np.transpose(dum_k))

                    Rel_Mat += damp_rate_l*dum_l+damp_rate_k*dum_k

    return Rel_Mat
    
    
    

def ZZ_relChanMatRep_Parallel(tc,coords,Nspins,gamma,basis,num_workers):
    def MatRep(w,An,Am):
        return MatRepLibParallel(w,An,Am,n_qubits=Nspins,num_workers=num_workers)
    
    Rel_Mat = np.zeros([len(basis),len(basis)],dtype=complex)

    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    damp_rate_0 = GammaRates(0,tc,coords[i],coords[j],coords[k],coords[l],gamma)
                    #damp_rate_diff = GammaRates(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)

                    dum_0 = -(8.0/3.0)*MatRep(basis,Sz(i)*Sz(j),Sz(k)*Sz(l))
                    #dum_0 += (2.0/3.0)*(MatRep(basis,S_plus(i)*S_minus(j),Sz(k)*Sz(l))+MatRep(basis,Sz(k)*Sz(l),S_minus(i)*S_plus(j)))
                    #dum_0 += np.conjugate(np.transpose(dum_0))
                    Rel_Mat+=damp_rate_0*dum_0

    #print("Testing using the correct ")
    return -0.25*Rel_Mat
