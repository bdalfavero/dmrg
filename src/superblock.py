import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from math import frexp, isqrt
import scipy.linalg as la

def make_superblock_hamiltonian(block1, block2, constants):
    """
    Make the superblock Hamiltonian for two block with spins added.
    block1, block2: dmrg blocks with the new spin at the left/right end
    constants: constants for the dot product J_x S^x_i S^x_(i+1)
    """

    # Make identity for block subspace
    eye_l = np.eye(block1.hamiltonian.shape[0])
    ham_ab = np.kron(block1.hamiltonian, eye_l) + np.kron(eye_l, block2.hamiltonian)
    for i in range(constants.size):
        ham_ab += constants[i] * np.kron(block1.end_ops[i], block2.end_ops[i])
    return sp.csr_matrix(ham_ab)


def dmrg_step(block1, block2, constants, size):
    """
    Perform a DMRG step in the infinite system algorithm.
    block1, block2: left and right blocks.
    constants: constants for exchange interaction between inner two spins.
    size: Maximum size of new basis.
    """

    # Make the superblock Hamiltonian, and get the ground state.
    hamiltonian_sb = make_superblock_hamiltonian(block1, block2, constants)
    w, v = eigsh(hamiltonian_sb, k=1)
    energy = w[0]
    psi0 = v[:, 0]
    # Reshape into the a_jk matrix and do the SVD to get the Schmidt basis.
    a_jk = psi0.reshape((block1.dim, block2.dim))
    u, s, vh = la.svd(a_jk)
    # Keep the M largest singular values.
    m = min(size, s.size)
    trans_op1 = u[:, 0:(m-1)]
    trans_op2 = vh.T[:, 0:(m-1)]
    # Truncate the block operators using the low-rank Schmidt basis.
    block1.truncate_operators(trans_op1)
    block2.truncate_operators(trans_op2)
    # Return the ground state energy and the ground state.
    return (energy, psi0)
