import sys
sys.path.append('./utils/')
import numpy as np
import scipy.io as spio
from scipy.linalg import expm
from matplotlib import pyplot as plt
import sys 
import openfermion as of
import pandas as pd

from analytical_fit import  get_chemical_shifts, Get_Det_And_Rates, RelMat_from_ops_and_rates_Eff_parallel
from basis_utils import read_spinach_info, build_list_ISTs, NormalizeBasis, build_symbolic_list_ISTs

text="""1      (0,0)   (0,0)   
  2      (0,0)   (1,1)   
  3      (0,0)   (1,0)   
  4      (0,0)   (1,-1)  
  5      (1,1)   (0,0)   
  6      (1,1)   (1,1)   
  7      (1,1)   (1,0)   
  8      (1,1)   (1,-1)  
  9      (1,0)   (0,0)   
  10     (1,0)   (1,1)   
  11     (1,0)   (1,0)   
  12     (1,0)   (1,-1)  
  13     (1,-1)  (0,0)   
  14     (1,-1)  (1,1)   
  15     (1,-1)  (1,0)   
  16     (1,-1)  (1,-1)  
"""

data = read_spinach_info(text)

basis = build_list_ISTs(data)
prefacts,Symb_basis = build_symbolic_list_ISTs(data)

#Normbasis = NormalizeBasis(basis,n_qubits=4,checkOrth=True) I have verified the orthonormalization of the basis
Normbasis = NormalizeBasis(basis,n_qubits=4,checkOrth=False)
Normbasis = np.array(Normbasis)

gammaF = 251814800
coord1 = np.array([-0.0551,-1.2087,-1.6523])*1e-10
coord2 = np.array([-0.8604 ,-2.3200 ,-0.0624])*1e-10

coords = np.array([coord1,coord2])

w1 = -376417768.6316 
w2 = -376411775.1523 
freqs = np.array([w1,w2])
tc = 0.5255e-9
B0 = 9.3933

zeeman_scalar_1 = -113.8796
zeeman_scalar_2 = -129.8002
zeeman_scalars = [zeeman_scalar_1,zeeman_scalar_2]

#w0*zeeman_scalars[i]/1e6
chem_shifts = get_chemical_shifts(gammaF,B0,zeeman_scalars)
Nspins = 2

list_jumps, list_damp_rates, list_dets=Get_Det_And_Rates(2*np.pi*freqs,tc,coords,Nspins,gammaF,chem_shifts)

#loading reference matrices...
loadMat = spio.loadmat('./data/DFG_secular.mat',squeeze_me=True)

Ham = loadMat['p']['H'].item()
R = loadMat['p']['R'].item()


#1) collect indices of non-vanishing entries of matrix...
thresh=1e-6
nonzero_idxs = []
for i in range(len(Normbasis)):
    for j in range(len(Normbasis)):
        if np.abs(R[i,j])>1e-6: 
            nonzero_idxs.append([i,j])

#jump_ops, rates, basis, Nspins, non_van_idxs, num_workers=None
if __name__ == '__main__':
    R_sec = RelMat_from_ops_and_rates_Eff_parallel(list_jumps,list_damp_rates,Normbasis,Nspins,nonzero_idxs,num_workers=2)

    print("Difference between reference relaxation matrix and the parallel-generated: ",np.linalg.norm(R_sec - R))






