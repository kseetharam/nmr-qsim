#For purposes of performance and cleanin-ness in coding with JAX, we consider some reformulation in the construction of Liouvillian in the Pauli basis..
from functools import partial
#from jax.experimental import sparse
from jax import lax
import scipy.linalg
from basis_utils import Sx,Sy,Sz
from basis_utils import MatRepLib, S_plus, S_minus
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import openfermion as of

def InnProd_jax(Op1,Op2):
    """ 
    Op1 and Op2 are JAX arrays
    """

    return jnp.trace(Op1.conj().T@Op2)


def commutator(A,B):
    """
    Returns: commutators between matrices A and B 
    """

    return A@B-B@A


def single_comm_superOp(observable,basis):
    """
    Returns: a matrix representation of the superoperator A in AO=[a,O] in a Pauli basis contained in the "basis" array 
    Args:
    observable, a matrix 
    basis, Pauli basis, array of matrices
    """
    def compute_element(i,j,observable=observable,basis=basis):
        #print("i,j:",i,j)
        basis_i = lax.dynamic_index_in_dim(basis, i, axis=0, keepdims=False)
        basis_j = lax.dynamic_index_in_dim(basis, j, axis=0, keepdims=False)

        return InnProd_jax(basis_i,commutator(observable,basis_j))

    N=len(basis)

    i_idx, j_idx = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()

    # Vectorized computation
    vec_compute = jax.vmap(compute_element)
    elements = vec_compute(i_flat, j_flat)

    # Reshape to matrix
    matrix = elements.reshape(N, N)

    return matrix

def double_comm_superop(outer_op,inn_op,basis):
    """
    Returns: a matrix representation of the superoperator A in AO=[outer_op,[inn_op,O] in a Pauli basis contained in the "basis" array 
    Args:
    outer_op, matrix representation for outer_op
    inn_op, matrix representation for inner op 
    basis, Pauli basis, array of matrices
    """

    def compute_element(i,j,outer_op=outer_op,inn_op=inn_op,basis=basis):
        #print("i,j:",i,j)
        basis_i = lax.dynamic_index_in_dim(basis, i, axis=0, keepdims=False)
        basis_j = lax.dynamic_index_in_dim(basis, j, axis=0, keepdims=False)

        return InnProd_jax(basis_i,commutator(outer_op(commutator(inn_op,basis_j))))

    N=len(basis)

    i_idx, j_idx = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()

    # Vectorized computation
    vec_compute = jax.vmap(compute_element)
    elements = vec_compute(i_flat, j_flat)

    # Reshape to matrix
    matrix = elements.reshape(N, N)

    return matrix

def Linb_chann_superop(An,Am,basis):
    """
    Returns: a matrix representation of the channel L_{An,Am}[rho]=An*rho*Am^{\dagger}-0.5*(An*Am^{\dagger}*rho+rho*An*Am^{\dagger} )
    Args: An, matrix form of a jump operator
    Am, matrix form of a jump operator
    basis, array of matrices, the Pauli basis used to represent the operators 
    """
    def compute_element(i,j,An=An,Am=Am,basis=basis):
        #print("i,j:",i,j)
        basis_i = lax.dynamic_index_in_dim(basis, i, axis=0, keepdims=False)
        basis_j = lax.dynamic_index_in_dim(basis, j, axis=0, keepdims=False)
        Am_dag = jnp.transpose(jnp.conjugate(Am))


        Linb_on_j = An@basis_j@Am_dag -0.5*(An@Am_dag@basis_j+basis_j@An@Am_dag)

        return InnProd_jax(basis_i,Linb_on_j)


    N=len(basis)

    i_idx, j_idx = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()

    # Vectorized computation
    vec_compute = jax.vmap(compute_element)
    elements = vec_compute(i_flat, j_flat)

    # Reshape to matrix
    matrix = elements.reshape(N, N)
    return matrix


def get_Heis_int_ops(Nspins):
    """
    Returns: a 1D array that contains the operators corresponding to Heisenberg interactions, retaining those with J couplings above min_coup in absolute value
    Args: 
    Nspins: number of spins
    """
    i, j = jnp.triu_indices(Nspins, k=1)
    List_ops = []
    for counter in range(len(i)):
        i_idx = i[counter]
        j_idx = j[counter]
        sp_op = of.get_sparse_operator(Sx(i_idx)*Sx(j_idx)+Sy(i_idx)*Sy(j_idx)+Sz(i_idx)*Sz(j_idx),n_qubits=Nspins)
        List_ops.append(sp_op.toarray())
    
    return jnp.array(List_ops)

def flat_upper_triang_mat(matrix):
    """
    Returns: a 1D array that results from flattening the upper triangular part of a matrix (excluding diagonal entries), useful to extract J couplings from Spinach's coupling matrix
    Args: matrix: a 2D array 
    """

    i, j = jnp.triu_indices(matrix.shape[0], k=1)

    # Extract and flatten
    strict_upper_flat = matrix[i, j]

    return strict_upper_flat

def get_H0_ops(Nspins):
    """
    Returns: a 1D array that contains the operators corresponding to H0
    Args: 
    Nspins: number of spins
    """
    Sz_s = []
    for i in range(Nspins):
        op = of.get_sparse_operator(Sz(i),n_qubits=Nspins)
        Sz_s.append(op.toarray())
    Sz_s = jnp.array(Sz_s)

    Heis_ints = get_Heis_int_ops(Nspins)

    return jnp.concatenate((Sz_s,Heis_ints))


@jax.jit
def build_H0_operator(theta, observables, basis):
    def term(t, M):
        Mk = single_comm_superOp(M, basis)
        return t * Mk

    # Vectorize over theta and observables
    return jax.vmap(term)(theta, observables).sum(axis=0)

@jax.jit
def exponentiate_H0(theta, observables, basis):

    return jax.scipy.linalg.expm(-1j*build_H0_operator(theta, observables, basis))


##### Functions dedicated to the construction of damping rates and jump operators...

def SpecFunc(w,tc):

    return 0.2*tc/(1 + w**2 * tc**2)

@jax.jit
def GammaRates_jax(w,tc,coord1,coord2,coord3,coord4,gamma):
    hbar = 1.054571628*1e-34
    diff1 = coord2-coord1
    #jax.debug.print("gamma^4 is={}",gamma**4)
    #jax.debug.print("diff between coord2 and coord1 ={}",diff1)

    diff2 = coord4 - coord3
    r1 = jnp.sqrt(jnp.dot(diff1,diff1))
    r2 = jnp.sqrt(jnp.dot(diff2,diff2))

    #jax.debug.print("r1 = {}", r1)
    #jax.debug.print("r2 = {}", r2)
    #jax.debug.print("diff1 ={}",diff1)
    #jax.debug.print("r2 = {}", r2)

    x1 = diff1[0]
    y1 = diff1[1]
    z1 = diff1[2]

    x2 = diff2[0]
    y2 = diff2[1]
    z2 = diff2[2]

    r1 = jnp.sqrt(x1**2+y1**2+z1**2)
    r2 = jnp.sqrt(x2**2+y2**2+z2**2)

    #jax.debug.print("x1 = {}", x1)

    #r1 = jnp.sqrt(x1**2+y1**2+z1**2)
    #r2 = jnp.sqrt(x2**2+y2**2+z2**2)

    Invariant = y2**2 * (2*y1**2-z1**2)-x2**2 * (y1**2+z1**2)+6*y1*y2*z1*z2-(y1**2-2*z1**2)*z2**2

    #jax.debug.print("Invariant1 = {}", Invariant)
    Invariant += 6*x1*x2*(y1*y2+z1*z2)+x1**2 * (2*x2**2-y2**2-z2**2)
    #jax.debug.print("Invariant2 = {}", Invariant)
    Invariant = (3/(r1**2 * r2**2))*Invariant
    #jax.debug.print("Invariant3 = {}", Invariant)
    
    Apref = Invariant

    #jax.debug.print("Apref = {}", Apref)
    #jax.debug.print("Specfunc vakues is = {}",SpecFunc(w,tc))

    #jax.debug.print("hbar^2 gamma^4/(r1^3 * r2^3) is={}",hbar**2 * gamma**4 /(r1**3 * r2**3))

    return hbar**2 * 1e-14*gamma**4 * SpecFunc(w,tc)*Apref/(r1**3 * r2**3)

####TODO: constructing the family of jump operators under the K_i classification is computationally inefficient
#but useful for debugging purposes. In a near-future version, we can generate the family of operators with 
#a single function and avoid redundant calculations


def K2_ops(Nspins,basis):
    """
    Returns: a jax array that contains the corresponding list of operators to the rates generated by the K2_rates_jax function 
    Args:
    Nspins: number of spins
    basis: array of matrices that encode the Pauli basis 
    """
    ops =[]
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    An = jnp.array(of.get_sparse_operator(S_plus(i)*S_plus(j),n_qubits=Nspins).toarray())
                    Am = jnp.array(of.get_sparse_operator(S_plus(k)*S_plus(l),n_qubits=Nspins).toarray())

                    dum= Linb_chann_superop(An,Am,basis)

                    An = jnp.array(of.get_sparse_operator(S_minus(k)*S_minus(l),n_qubits=Nspins).toarray())
                    Am = jnp.array(of.get_sparse_operator(S_minus(i)*S_minus(j),n_qubits=Nspins).toarray())

                    dum+= Linb_chann_superop(An,Am,basis)

                    dum+=jnp.conjugate(jnp.transpose(dum))

                    ops.append(dum)

    return jnp.array(ops)


def K1_ops(Nspins,basis):
    """
    Returns: two jax arrays that contains the corresponding list of operators to the rates generated by the K1_rates_jax function 
    Args:
    Nspins: number of spins
    basis: array of matrices that encode the Pauli basis 
    """
    def mat_wrap(of_op,Nspins=Nspins):
        return jnp.array(of.get_sparse_operator(of_op,n_qubits=Nspins).toarray())

    #ops_l =[]
    #ops_k =[]

    #ops = jnp.array([])
    ops = []
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    
                    dum_l=Linb_chann_superop(mat_wrap(Sz(i)*S_plus(j)),mat_wrap(Sz(k)*S_plus(l)),basis)
                    dum_l+=Linb_chann_superop(mat_wrap(Sz(k)*S_minus(l)),mat_wrap(Sz(i)*S_minus(j)),basis)
                    dum_l+=Linb_chann_superop(mat_wrap(S_plus(i)*Sz(j)),mat_wrap(Sz(k)*S_plus(l)),basis)
                    dum_l+=Linb_chann_superop(mat_wrap(Sz(k)*S_minus(l)),mat_wrap(S_minus(i)*Sz(j)),basis)
                    dum_l+=jnp.conjugate(jnp.transpose(dum_l))
                    #ops_l.append(dum_l)

                    dum_k = Linb_chann_superop(mat_wrap(S_plus(i)*Sz(j)),mat_wrap(S_plus(k)*Sz(l)),basis)
                    dum_k+= Linb_chann_superop(mat_wrap(S_minus(k)*Sz(l)),mat_wrap(S_minus(i)*Sz(j)),basis)
                    dum_k+= Linb_chann_superop(mat_wrap(Sz(i)*S_plus(j)),mat_wrap(S_plus(k)*Sz(l)),basis)
                    dum_k+= Linb_chann_superop(mat_wrap(S_minus(k)*Sz(l)),mat_wrap(Sz(i)*S_minus(j)),basis)
                    dum_k+=jnp.conjugate(jnp.transpose(dum_k))
                    #ops_k.append(dum_k)
                    #ops = jnp.concatenate([ops,jnp.array([dum_l,dum_k])])
                    ops.append(dum_l)
                    ops.append(dum_k)

                    


    return jnp.array(ops)


def K0_ops(Nspins,basis):
    """
    Returns: two jax arrays that contains the corresponding list of operators to the rates generated by the K1_rates_jax function 
    Args:
    Nspins: number of spins
    basis: array of matrices that encode the Pauli basis 
    """
    def mat_wrap(of_op,Nspins=Nspins):
        return jnp.array(of.get_sparse_operator(of_op,n_qubits=Nspins).toarray())
    
    #ops_0 =[]
    #ops_diff =[]
    #ops = jnp.array([])
    ops = []
    for i in range(Nspins):
        for j in range(i+1,Nspins):
            
            for k in range(Nspins):
                for l in range(k+1,Nspins):
                    dum_0 = (8.0/3.0)*Linb_chann_superop(mat_wrap(Sz(i)*Sz(j)),mat_wrap(Sz(k)*Sz(l)),basis)
                    dum_0+= -(2.0/3.0)*Linb_chann_superop(mat_wrap(S_plus(i)*S_minus(j)),mat_wrap(Sz(k)*Sz(l)),basis)
                    dum_0+= -(2.0/3.0)*Linb_chann_superop(mat_wrap(Sz(k)*Sz(l)),mat_wrap(S_minus(i)*S_plus(j)),basis)
                    dum_0 += jnp.conjugate(jnp.transpose(dum_0))

                    #ops_0.append(-1.0*dum_0)

                    dum_diff = -(2.0/3.0)*Linb_chann_superop(mat_wrap(Sz(i)*Sz(j)),mat_wrap(S_plus(k)*S_minus(l)),basis)
                    dum_diff+= -(2.0/3.0)*Linb_chann_superop(mat_wrap(S_minus(k)*S_plus(l)),mat_wrap(Sz(i)*Sz(j)),basis)
                    dum_diff+=  (1.0/6.0)*Linb_chann_superop(mat_wrap(S_plus(i)*S_minus(j)),mat_wrap(S_plus(k)*S_minus(l)),basis)
                    dum_diff+= (1.0/6.0)*Linb_chann_superop(mat_wrap(S_minus(k)*S_plus(l)),mat_wrap(S_minus(i)*S_plus(j)),basis)
                    dum_diff+= (1.0/6.0)*Linb_chann_superop(mat_wrap(S_minus(i)*S_plus(j)),mat_wrap(S_plus(k)*S_minus(l)),basis)
                    dum_diff+= (1.0/6.0)*Linb_chann_superop(mat_wrap(S_minus(k)*S_plus(l)),mat_wrap(S_plus(i)*S_minus(j)),basis)
                    dum_diff+= jnp.conjugate(jnp.matrix_transpose(dum_diff))
                    #ops_diff.append(-1.0*dum_diff)
                    #ops = jnp.concatenate([ops,[dum_0, dum_diff]])
                    ops.append(dum_0)
                    ops.append(dum_diff)


    return jnp.array(ops)

@partial(jax.jit, static_argnames=['Nspins'])
def K2_rates_jax(freqs,tc,coords,Nspins,gamma):
    """
    Returns: 1) jnp array of K2 damping rates
    Args:
    freqs: array that contains the Zeeman frequencies of the N spins
    tc: bath correlation time
    coords: coordinates of spins
    Nspins: number of spins
    gamma: gyromagnetic ratio for spins (assuming homonuclear for now)
    basis: Pauli basis for matrix representation of operators, it needs to be an array of matrices
    """
    
    def get_rate(i,j,k,l,freqs=freqs,tc=tc,coords=coords,gamma=gamma):

        damp_rate = GammaRates_jax(freqs[k]+freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
        #jax.debug.print("Gamma = {}", damp_rate)

        return damp_rate
    
    
    vec_get_rate = jax.vmap(lambda x: get_rate(x[0], x[1], x[2], x[3]))
    #vec_get_op = jax.vmap(lambda x: get_op(x[0],x[1],x[2],x[3]))
    
    ij = jnp.array([(i, j) for i in range(Nspins) for j in range(i + 1, Nspins)])
    kl = jnp.array([(k, l) for k in range(Nspins) for l in range(k + 1, Nspins)])

    # Create the full 2D grid of combinations between ij and kl
    ij_grid = jnp.repeat(ij, len(kl), axis=0)
    kl_grid = jnp.tile(kl, (len(ij), 1))

    # Combine into a single array of shape 
    ijkl = jnp.hstack([ij_grid, kl_grid])

    
    rates = vec_get_rate(ijkl)
    #ops = vec_get_op(ijkl)
    #print("Rates are:", rates)

    return rates

@partial(jax.jit, static_argnames=['Nspins'])
def K1_rates_jax(freqs,tc,coords,Nspins,gamma):
    """
    Returns: 1) jnp array of K1_l damping rates and 2) array of K1_k damping rates
    Args:
    freqs: array that contains the Zeeman frequencies of the N spins
    tc: bath correlation time
    coords: coordinates of spins
    Nspins: number of spins
    gamma: gyromagnetic ratio for spins (assuming homonuclear for now)
    basis: Pauli basis for matrix representation of operators, it needs to be an array of matrices
    """
    def get_rate(i,j,k,l,freqs=freqs,tc=tc,coords=coords,gamma=gamma):
        rate_l = GammaRates_jax(freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
        rate_k = GammaRates_jax(freqs[k],tc,coords[i],coords[j],coords[k],coords[l],gamma)
        #jax.debug.print("Gamma = {}", rate_l)
        return rate_l, rate_k
      
    
    vec_get_rate = jax.vmap(lambda x: get_rate(x[0], x[1], x[2], x[3]))
    #vec_get_op = jax.vmap(lambda x: get_op(x[0],x[1],x[2],x[3]))
    
    ij = jnp.array([(i, j) for i in range(Nspins) for j in range(i + 1, Nspins)])
    kl = jnp.array([(k, l) for k in range(Nspins) for l in range(k + 1, Nspins)])

    # Create the full 2D grid of combinations between ij and kl
    ij_grid = jnp.repeat(ij, len(kl), axis=0)
    kl_grid = jnp.tile(kl, (len(ij), 1))

    # Combine into a single array of shape (num_combinations, 4)
    ijkl = jnp.hstack([ij_grid, kl_grid])

    
    rates = vec_get_rate(ijkl)
    return jnp.concatenate(rates)

@partial(jax.jit, static_argnames=['Nspins'])
def K0_rates_jax(freqs,tc,coords,Nspins,gamma):
    """
    Returns: 1) jnp array of K0_0 damping rates and 2) array of K0_diff damping rates
    Args:
    freqs: array that contains the Zeeman frequencies of the N spins
    tc: bath correlation time
    coords: coordinates of spins
    Nspins: number of spins
    gamma: gyromagnetic ratio for spins (assuming homonuclear for now)
    basis: Pauli basis for matrix representation of operators, it needs to be an array of matrices
    """
    def get_rate(i,j,k,l,freqs=freqs,tc=tc,coords=coords,gamma=gamma):
        rate_0 = GammaRates_jax(0.0,tc,coords[i],coords[j],coords[k],coords[l],gamma)
        rate_diff = GammaRates_jax(freqs[k]-freqs[l],tc,coords[i],coords[j],coords[k],coords[l],gamma)
        #jax.debug.print("Gamma = {}", rate_l)
        return rate_0, rate_diff
      
    
    vec_get_rate = jax.vmap(lambda x: get_rate(x[0], x[1], x[2], x[3]))
    #vec_get_op = jax.vmap(lambda x: get_op(x[0],x[1],x[2],x[3]))
    
    ij = jnp.array([(i, j) for i in range(Nspins) for j in range(i + 1, Nspins)])
    kl = jnp.array([(k, l) for k in range(Nspins) for l in range(k + 1, Nspins)])

    # Create the full 2D grid of combinations between ij and kl
    ij_grid = jnp.repeat(ij, len(kl), axis=0)
    kl_grid = jnp.tile(kl, (len(ij), 1))

    # Combine into a single array of shape (num_combinations, 4)
    ijkl = jnp.hstack([ij_grid, kl_grid])

    
    rates = vec_get_rate(ijkl)
    return jnp.concatenate(rates)



@partial(jax.jit,static_argnames=['Nspins'])
def build_R_operator(freqs,tc,coords,gamma,basis,Nspins):
    """
    Returns: \sum_{i}\gamma_{i}O_{i}, \gamma_{i}'s being scalars and O_{i}'s being operators 
    """

    #get rates and operators...
    k2_rates = K2_rates_jax(freqs,tc,coords,Nspins,gamma)
    k1_rates = K1_rates_jax(freqs,tc,coords,Nspins,gamma)
    k0_rates = K0_rates_jax(freqs,tc,coords,Nspins,gamma)

    k2_ops = K2_ops(Nspins,basis)
    k1_ops = K1_ops(Nspins,basis)
    k0_ops = K0_ops(Nspins,basis)

    rates =0.25*jnp.concatenate([k2_rates,k1_rates,k0_rates])
    ops = jnp.concatenate([k2_ops,k1_ops,k0_ops])

    #return jnp.tensordot(k2_rates,k2_ops,axes=1),jnp.tensordot(k1_rates,k1_ops,axes=1),jnp.tensordot(k0_rates,k0_ops,axes=1),jnp.tensordot(rates, ops, axes=1)
    return jnp.tensordot(rates, ops, axes=1)

###Generator of the total Liouvillian and the time-evolution operator...
@partial(jax.jit,static_argnames=['Nspins'])
def get_shifted_zeem_freqs(freqs,gamma,B0,offset,Nspins):
    """
    Returns: shifted Zeeman angular frequencies of spins
    Args:
    freqs: frequencies (in Hz) of the spins
    gamma: gyromagnetic ratio of spins (assuming homonuclear scenario)
    B0: Magnetic field in Teslas
    offset: frequency offset, in Hz
    Nspins: number of spins 
    """
    omega0_shift = gamma*B0+2*jnp.pi*offset
    omegas = []
    for i in range(Nspins):
        omegas.append(2*jnp.pi*freqs[i]+omega0_shift)

    return jnp.array(omegas)


@partial(jax.jit,static_argnames=['Nspins'])
def build_time_evol_op(time,freqs,Jcouplings,B0,freq_offset,tc,coords,gamma,coh_observables,basis,Nspins):
    """
    Returns: exponential of Liouvillian in a Pauli basis, defined by the basis argument
    Args:
    time: time of propagation in seconds
    freqs: frequencies (in Hz) of Zeeman-split spins
    Jcouplings: 1D array of two-body coherent couplings 
    B0: strenght of magnetic field (in Teslas)
    freq_offset: frequency offset (in Hz) for spins
    tc: bath correlation time in seconds
    coords: array of spin coordinates
    gamma: gyromagetic ratio of spins (homonuclear case for now)
    coh_observables: array of the the operators that compose the coherent Hamiltonian
    basis: array that contains the Pauli basis
    Nspins: number of spins
    """
    ###Get the parameters for construction of coherent Hamiltonian...
    omegas = get_shifted_zeem_freqs(freqs,gamma,B0,freq_offset,Nspins)
    coh_params = jnp.concatenate((omegas,2*jnp.pi*Jcouplings)) #Make sure ordering coincides with that of coh_observables!
    
    R = build_R_operator(2*jnp.pi*freqs,tc,coords,gamma,basis,Nspins)

    H0 = build_H0_operator(coh_params, coh_observables, basis)

    return jax.scipy.linalg.expm(-1j*time*H0+time*R)
    #return H0,R



