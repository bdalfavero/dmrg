import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from math import frexp, isqrt
import scipy.linalg as la
import block

class DMRGSystem:

    def __init__(self, ham0, end_ops, new_ops, constants, max_size):
        """
        Initialize the system.
        ham0 - Hamiltonian of starting blocks.
        end_ops - tuple with (end_ops_left, end_ops_right), each a list of operators.
        new_ops - Operators added to the end of a block.
        constants - Coupling constants for spin-spin interactions.
        max_size - Max basis size on truncation.
        """

        # Set up initial blocks.
        self.constants = constants
        self.block1 = block.Block(ham0, end_ops[0], self.constants)
        self.block2 = block.Block(ham0, end_ops[1], self.constants)
        self.new_ops = new_ops
        self.max_size = max_size

    def make_superblock_hamiltonian(self):
        """
        Make the superblock Hamiltonian for two block with spins added.
        block1, block2: dmrg blocks with the new spin at the left/right end
        constants: constants for the dot product J_x S^x_i S^x_(i+1)
        """

        # Make identity for block subspace
        eye_l = np.eye(self.block1.hamiltonian.shape[0])
        ham_ab = np.kron(self.block1.hamiltonian, eye_l) + np.kron(eye_l, self.block2.hamiltonian)
        for i in range(self.constants.size):
            ham_ab += self.constants[i] * np.kron(self.block1.end_ops[i], self.block2.end_ops[i])
        return sp.csr_matrix(ham_ab)

    def finite_dmrg_step(self):
        """
        Perform a DMRG step in the infinite system algorithm.
        block1, block2: left and right blocks.
        constants: constants for exchange interaction between inner two spins.
        size: Maximum size of new basis.
        """

        # Make the superblock Hamiltonian, and get the ground state.
        hamiltonian_sb = self.make_superblock_hamiltonian()
        w, v = eigsh(hamiltonian_sb, k=1)
        energy = w[0]
        psi0 = v[:, 0]
        # Reshape into the a_jk matrix and do the SVD to get the Schmidt basis.
        a_jk = psi0.reshape((self.block1.dim, self.block2.dim))
        u, s, vh = la.svd(a_jk)
        # Keep the M largest singular values.
        m = min(self.max_size, s.size)
        trans_op1 = u[:, 0:(m-1)]
        trans_op2 = vh.T[:, 0:(m-1)]
        # Truncate the block operators using the low-rank Schmidt basis.
        self.block1.truncate_operators(trans_op1)
        self.block2.truncate_operators(trans_op2)
        # Return the ground state energy and the ground state.
        return (energy, psi0)
    
    def finite_dmrg_algorithm(self, steps):
        """
        Perform a number of steps for the finite DMRG algorithm.
        steps - The number of steps to take.
        """

        energies = np.zeros(steps)
        for i in range(steps):
            self.block1.add_spin(self.new_ops, "right")
            self.block2.add_spin(self.new_ops, "left")
            energy, psi0 = self.finite_dmrg_step()
            length = self.block1.size + self.block2.size
            energies[i] = energy / float(length)
        return energies
