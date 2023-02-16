#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from math import frexp, isqrt
import block
import superblock as sb

# Pauli matrices
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_y = np.array([[0.0, 1j], [-1j, 0.0]])
id2 = np.eye(2)

def two_spin_hamiltonian(j):
    # Return the Hamiltonian for 2 spins
    # in the default basis.
    xx = np.kron(sigma_x, sigma_x)
    yy = np.kron(sigma_y, sigma_y)
    zz = np.kron(sigma_z, sigma_z)
    return j * (xx + yy + zz)

def end_ops(dir):
    # Initial values of the end-of-block operators
    # when the system is two spins in the default
    # basis.
    # dir - either "left" or "right", the side with the end.
    if dir == "right": 
        return [np.kron(id2, sigma_x), 
                np.kron(id2, sigma_y), 
                np.kron(id2, sigma_z)]
    else:
        return [np.kron(sigma_x, id2), 
                np.kron(sigma_y, id2), 
                np.kron(sigma_z, id2)]


# Set up initial system.
ham2 = two_spin_hamiltonian(1.0)
j_xyz = np.ones(3)
new_ops = [sigma_x, sigma_y, sigma_z]
end_ops_lr = (end_ops("left"), end_ops("right"))
system = sb.DMRGSystem(ham2, end_ops_lr, new_ops, j_xyz, 20)

system.finite_dmrg_algorithm(30)
