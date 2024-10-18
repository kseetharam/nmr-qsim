import numpy as np
import openfermion as of
from openfermion import QubitOperator
import re
import scipy.io as spio


###We start by defining the IST basis. TODO: for the two-spin system this can be done "by hand", we can implement a strategy to build this basis from spinach "summary" output
def S_plus(i):
    
    return 0.5*(QubitOperator('X'+str(i))+1j*QubitOperator('Y'+str(i)))

def S_minus(i):
    return 0.5*(QubitOperator('X'+str(i))-1j*QubitOperator('Y'+str(i)))

def Sz(i):
    return 0.5*QubitOperator('Z'+str(i))

def Sx(i):
    return 0.5*QubitOperator('X'+str(i))

def Sy(i):
    return 0.5*QubitOperator('Y'+str(i))

def T1_0(i):
    return Sz(i)

def T1_1(i):
    return (-1.0/np.sqrt(2))*S_plus(i)

def T1_min1(i):
    return (1.0/np.sqrt(2)*S_minus(i))
def T2_2(i):
    return 0.5*S_plus(i)*S_plus(i)
def T2_min2(i):
    return 0.5*S_minus(i)*S_minus(i)
def T2_1(i):
    return -0.5*(Sz(i)*S_plus(i)+S_plus(i)*Sz(i))

def T2_min1(i):
    return 0.5*(Sz(i)*S_minus(i)+S_minus(i)*Sz(i))

def T2_0(i):
    return np.sqrt(2.0/3.0)*(Sz(i)*Sz(i)-0.25*(S_plus(i)*S_minus(i)+S_minus(i)*S_plus(i)))

def read_spinach_info(text):
    lines = text.splitlines()

    # Extract integers from each line
    extracted_data = []

    for line in lines:
        # Find all integers in each line
        numbers = list(map(int, re.findall(r'-?\d+', line)))
        extracted_data.append(numbers)
    return extracted_data

def build_list_ISTs(data):
    """
    From data extracted from text, build the list of the ISTs 
    """
    nspins = (len(data[0])-1)//2
    basis = []
    for element in data:
        ISTOp = 1
        for i in range(nspins):
            #print(i)
            Op = buildIST(element[1+2*i],element[2+2*i],i)
            #print("Operator is: ", Op)
            if Op==None:
                print("We obtain no element for element:", i)
                print(element)
            ISTOp=ISTOp*Op
        basis.append(ISTOp)

    return basis
    
#TODO: we get zeros for the 2-rank tensors. we need to undertand what is going on there... It turns out that for the 2 spin system this does not matter...

def buildIST(l,m,i):
    """
    Build the (l,m) IST for the i-th spin. It turns out that for dipolar interactions, we need to consider l=0,1,2
    """
    if l==0 and m==0:
        return QubitOperator([])
    elif l==1 and m==0:
        return T1_0(i)
    elif l==1 and m==-1:
        return T1_min1(i)
    elif l==1 and m==1:
        return T1_1(i)
    elif l==2 and m==0:
        return T2_0(i)
    elif l==2 and m==-2:
        return T2_min2(i)
    elif l==2 and m==-1:
        return T2_min1(i)
    elif l==2 and m==1:
        return T2_1(i)
    elif l==2 and m==2:
        return T2_2(i)
    
###Importantly, the dipolar relaxation matrix can be described in terms of a rank2 two-body ISTs, which we define below:
def T2_0ij(i,j):
    sum = Sz(i)*Sz(j)-0.25*(S_plus(i)*S_minus(j)+S_minus(i)*S_plus(j))
    return np.sqrt(2.0/3)*sum

def T2_min2ij(i,j):
    return 0.5*S_minus(i)*S_minus(j)

def T2_min1ij(i,j):
    return 0.5*(Sz(i)*S_minus(j)+S_minus(i)*Sz(j))

def T2_1ij(i,j):
    return -0.5*(Sz(i)*S_plus(j)+S_plus(i)*Sz(j))

def T2_2ij(i,j):
    return 0.5*S_plus(i)*S_plus(j)

#Wrapper for the construction of the rank-2 tensors:
def T2_ij(i,j,m):
    if m ==0:
        return T2_0ij(i,j)
    elif m == -2:
        return T2_min2ij(i,j)
    elif m == -1:
        return T2_min1ij(i,j)
    elif m == 1:
        return T2_1ij(i,j)
    elif m == 2:
        return T2_2ij(i,j)

def commRank2(m,nspins,op):
    """
    Compute the commutator of op with the sum of rank2 operators with m number

    """
    Comm = QubitOperator()

    for i in range(nspins):
        for j in range(nspins):
            Comm += of.commutator(T2_ij(i,j,m),op)

    return Comm

def InnProd(Op1,Op2,n_qubits=None):
    spOp1 = of.get_sparse_operator(Op1,n_qubits=n_qubits)
    spOp2 = of.get_sparse_operator(Op2,n_qubits=n_qubits)

    return np.trace(spOp1.toarray().conj().T@spOp2.toarray())


def NormalizeBasis(basis,n_qubits=None,checkOrth=True):
    """
    Function that normalizes a given basis. IST basis is not normalized
    """
    NormBas = []
    for i in range(len(basis)):
        #spMat = of.get_sparse_operator(basis[i],n_qubits=n_qubits)
        #innprod = np.trace(spMat.toarray().conj().T@spMat.toarray())
        innprod = InnProd(basis[i],basis[i],n_qubits=n_qubits)

        NormBas.append(np.sqrt(1.0/innprod)*basis[i])

    if checkOrth:
        for i in range(len(basis)):
            for j in range(i+1,len(basis)):
                test = InnProd(NormBas[i],NormBas[j],n_qubits=n_qubits)
                if test != 0.0:
                    print("Warning, inner product of different basis elements is: ",test)
    
    return NormBas


def buildStrucMat(basis,nspins):
    
    NBasis = len(basis)

    Mat_Op = np.zeros([NBasis,NBasis],dtype=complex)
    #Two outer-loop indexes for the matrix relaxation of the operator
    for i in range(NBasis):
        for j in range(NBasis):

            #Two inner-loop for commutator construction
            R_op_on_basis = QubitOperator() 
            for m in range(-2,3):
                for m1 in range(-2,3):
                    #TODO: verify the hermitian conjugation on one of the operators here...
                    inner_comm = commRank2(m,nspins,basis[i])#of.commutator(basis[l],basis[j])
                    outer_comm = commRank2(m1,nspins,inner_comm)
                    R_op_on_basis+=outer_comm
            ##Inner product computation:
            Prod = basis[i]*R_op_on_basis
            sp_Prod = of.get_sparse_operator(Prod)

            inner_prod=(1/2**nspins)*np.trace(sp_Prod.toarray())

            Mat_Op[i,j] = inner_prod

    return Mat_Op


def Linb_Channel(An,Am,rho,gamma=1.0):
    """
    Apply the Linbladian with the jump operators An and Am^\dagger to operator rho, with the rate gamma
    """
    return gamma*(An*rho*of.hermitian_conjugated(Am)-0.5*(An*of.hermitian_conjugated(Am)*rho+rho*An*of.hermitian_conjugated(Am)))


def MatRepLib(basis,An,Am,n_qubits=2):

    Nbasis = len(basis)
    MatRep = np.zeros([Nbasis,Nbasis],dtype=complex)

    for i in range(Nbasis):
        for j in range(Nbasis):
            MatRep[i,j] = InnProd(basis[i],Linb_Channel(An,Am,basis[j]),n_qubits=n_qubits)
    
    return MatRep

def MatRepCommChannel(basis,An,Am,n_qubits=2):
    """
    Returns a matrix representation of the superoperator [An[Am,]], in the basis "basis"
    """
    Nbasis = len(basis)
    MatRep = np.zeros([Nbasis,Nbasis],dtype=complex)

    for i in range(Nbasis):
        for j in range(Nbasis):
            MatRep[i,j] = InnProd(basis[i],of.commutator(An,of.commutator(Am,basis[j])),n_qubits=n_qubits)
    
    return MatRep




def gram_schmidt_ops(ops):
    """
    Perform Gram-Schmidt orthogonalization on a set of operators.

    Args:
    vectors (list of np.ndarray): List of input vectors to be orthogonalized.

    Returns:
    np.ndarray: Orthonormal basis for the subspace spanned by the input vectors.
    """
    orthonormal_basis = []
    TransMat = np.zeros([len(ops),len(ops)],dtype=complex)
    
    counter = 0 

    row_idx = 0
    #col_idx = 0
    for v in ops:
        count_orth = 0

        if row_idx==0:
            TransMat[row_idx,row_idx] = 1.0

        # Orthogonalize v against the previously computed basis vectors
        for u in orthonormal_basis:
            #print("Row idx is:", row_idx)
            #if row_idx==0:
                #print("Entered here")
            #    v = v - HB_norm(v, u) * u
            #    print("I changed the zeroth element")
            #    TransMat[row_idx,row_idx] = 1.0
            #else:
                #print("Entered there")
            u_norm = HB_norm(u,u)
            if u_norm> 1e-10:
                coeff = HB_norm(v, u)/HB_norm(u,u)
                v = v - coeff * u
            else:
                v = v -  HB_norm(v, u)
            #For the orthogonal transformation...
            TransMat[row_idx,row_idx] = 1.0
            TransMat[row_idx,:]=TransMat[row_idx,:]-coeff*TransMat[count_orth,:]

            count_orth+=1
        
        # Normalize the orthogonalized vector
        norm_v = np.sqrt(HB_norm(v,v))
        if norm_v > 1e-10:  # Avoid division by zero for nearly zero vectors
            orthonormal_basis.append(v *(1/norm_v))
            TransMat[row_idx,:] = (1.0/norm_v)*TransMat[row_idx,:]
        #else:
        #    orthonormal_basis.append(v)

            #counter+=1 #This counts the number of orthogonalized vectors so far...
        
            row_idx+=1
        else:
            orthonormal_basis.append(np.zeros_like(ops[0]))
            TransMat[row_idx,:] = np.zeros(len(TransMat[row_idx,:]))
            row_idx+=1
    
    return orthonormal_basis,TransMat



def HB_norm(mat1,mat2):
    return np.trace(mat1.conj().T@mat2)



def BuildChannMat(basis,An,Am,gamma=1.0):

    Mat = np.zeros([len(basis),len(basis)])

    for i in range(len(basis)):
        for j in range(len(basis)):
            Mat[i,j] = InnProd(basis[i],Linb_Channel(An,Am,basis[j],gamma=gamma),n_qubits=2)

    return Mat

def DiagElofSysChannel(op,Filtbasis,Filtms):

    Mat = np.zeros([len(Filtbasis),len(Filtbasis)])

    for i in range(len(Filtbasis)):
        for j in range(len(Filtbasis)):
            if np.abs(Filtms[i]-Filtms[j])==0:
                Mat[i,j] = InnProd(op,Linb_Channel(Filtbasis[i],Filtbasis[j],op),n_qubits=2)


    return Mat 

def DiagSystEqs(indexes,basis,Filtbasis,Rmat):

    Mat = np.zeros([len(indexes),len(Filtbasis)])
    Vect = np.zeros(len(indexes))

    for i in range(len(Filtbasis)):
        for j in range(len(indexes)):
            Mat[j,i] =InnProd(basis[indexes[j]],Linb_Channel(Filtbasis[i],Filtbasis[i],basis[indexes[j]]),n_qubits=2)

    for j in range(len(indexes)):
        Vect[i]=Rmat[indexes[j],indexes[j]]

    return Mat, Vect

def BuiltSystEqs(basis,BasisCols,Filtms,idxs_BasisRows,Rmat):
    """
    Args:
    basis: the array that contains the list of all operators used in the model
    BasisCols: array that contains the sample of operators to build the matrix of coefficients along the columns
    Filtms: contains the values of m corresponding to each operator in BasisCols
    idxs_BasisRows: a list that contains the indexes that label each of the elements on BasisRows, in the original list of operators
    Rmat, the relaxation matrix
    """

    BasisRows = basis[idxs_BasisRows]

    n_rows = len(BasisRows)
    n_cols = len(BasisCols)

    comp_idxs_rows = []
    comp_idxs_cols = []
    for i in range(n_rows):
        for j in range(n_rows):
            comp_idxs_rows.append([i,j])

    for i in range(n_cols):
        for j in range(n_cols):
            comp_idxs_cols.append([i,j])

    size_rows = len(comp_idxs_rows)
    size_cols = len(comp_idxs_cols)

    SupMatA = np.zeros([size_rows,size_cols],dtype=complex)
    for i in range(size_rows):
        row_idx1 = comp_idxs_rows[i][0]
        row_idx2 = comp_idxs_rows[i][1]
        for j in range(size_cols):
            col_idx1 = comp_idxs_cols[j][0]
            col_idx2 = comp_idxs_cols[j][1]

            if np.abs(Filtms[col_idx1]-Filtms[col_idx2])==0:
                SupMatA[i,j] = InnProd(BasisRows[row_idx1],Linb_Channel(BasisCols[col_idx1],BasisCols[col_idx2],BasisRows[row_idx2]),n_qubits=2)


####Extracting the coefficients of the R matrix...
    CVect = np.zeros(size_rows,dtype=complex)

    counter = 0
    for i in range(n_rows):
        for j in range(n_rows):
            row_idx = idxs_BasisRows[i]
            col_idx = idxs_BasisRows[j]
            CVect[counter] = Rmat[row_idx,col_idx]
            counter+=1

    return SupMatA, CVect

def extractij_fromR(index,dimR):
    row = int(np.floor(index/dimR))
    col = index%dimR
    return row,col


def DecomposeTarget(OrthBasis,target):
    """
    Obtain the weights of a orthonormalized matrix basis such the target matrix is expressed as a linear combination of this basis
    Returns:
    the array of weights of each element of the orthonormal basis to reproduce the target matrix target
    """
    coeffs = []
    for i in range(len(OrthBasis)):
        coeffs.append(HB_norm(OrthBasis[i],target))

    coeffs = np.array(coeffs)

    return coeffs



def GetOrigLinbs(OrthBasis,TransMat,target):

    weightsOrig = np.copy(TransMat)

    coeffs = DecomposeTarget(OrthBasis,target)

    counter=0
    for i in range(len(TransMat[:,0])):
        weightsOrig[i,:]=coeffs[i]*weightsOrig[i,:]
        counter+=1

    totweightOrig = []
    for i in range(len(weightsOrig[0,:])):
        totweightOrig.append(np.sum(weightsOrig[:,i]))

    totweightOrig = np.array(totweightOrig)
    sort_idxs = (-1*np.abs(totweightOrig)).argsort()

    return totweightOrig,sort_idxs


def MatRepSimpleCom(basis,op):

    Nbasis=len(basis)
    Matrix = np.zeros([Nbasis,Nbasis],dtype=complex)

    for i in range(Nbasis):
        for j in range(Nbasis):
            Matrix[i,j] = InnProd(basis[i],of.commutator(op,basis[j]),n_qubits=2)

    return Matrix



