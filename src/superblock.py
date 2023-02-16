import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from math import frexp, isqrt
import scipy.linalg as la
import block
import copy

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
        # Set up the history. This is a dictionary with keys
        # "left" and "right". Each key points to a dictionary
        # mapping block sizes to blocks.
        self.history = {"left": {}, "right": {}}

    def make_superblock_hamiltonian(self):
        """
        Make the superblock Hamiltonian for two block with spins added.
        block1, block2: dmrg blocks with the new spin at the left/right end
        constants: constants for the dot product J_x S^x_i S^x_(i+1)
        """

        # TODO: This function is built for the infinite algorithm.
        # Adapt it to the finite algorithm! Block dimensions can differ now.
        
        # Make identity for block subspace
        eye_l = np.eye(self.block1.hamiltonian.shape[0])
        ham_ab = np.kron(self.block1.hamiltonian, eye_l) + np.kron(eye_l, self.block2.hamiltonian)
        for i in range(self.constants.size):
            ham_ab += self.constants[i] * np.kron(self.block1.end_ops[i], self.block2.end_ops[i])
        return sp.csr_matrix(ham_ab)
    
    def make_truncation_operators(self, psi0):
        """
        Make two operators for truncating the basis of both blocks.
        """

        # Reshape into the a_jk matrix and do the SVD to get the Schmidt basis.
        a_jk = psi0.reshape((self.block1.dim, self.block2.dim))
        u, s, vh = la.svd(a_jk)
        # Keep the M largest singular values.
        m = min(self.max_size, s.size)
        trans_op1 = u[:, 0:(m-1)]
        trans_op2 = vh.T[:, 0:(m-1)]
        return (trans_op1, trans_op2)

    def infinite_dmrg_step(self):
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
        # Get the truncation operators.
        trans_op1, trans_op2 = self.make_truncation_operators(psi0)
        # Truncate the block operators using the low-rank Schmidt basis.
        self.block1.truncate_operators(trans_op1)
        self.block2.truncate_operators(trans_op2)
        # Return the ground state energy and the ground state.
        return (energy, psi0)
    
    def infinite_dmrg_algorithm(self, steps):
        """
        Perform a number of steps for the finite DMRG algorithm.
        steps - The number of steps to take.
        """

        energies = np.zeros(steps)
        for i in range(steps):
            # Add a spin to each block and get the energy and ground state.
            self.block1.add_spin(self.new_ops, "right")
            self.block2.add_spin(self.new_ops, "left")
            energy, psi0 = self.infinite_dmrg_step()
            # Store the current energy/length.
            length = self.block1.size + self.block2.size
            energies[i] = energy / float(length)
            # Add these blocks to the history.
            self.history["left"][self.block1.size] = copy.copy(self.block1)
            self.history["right"][self.block2.size] = copy.copy(self.block2)

        return energies
    
    def finite_dmrg_step(self, dir):
        """
        Take a DMRG step with the finite algorithm. Shift the split
        between the blocks either to the left or right. Then diagonalize and truncate.
        Parameters:
        dir - String, either "left" or "right".
        """

        # Shift a spin from one block to another.
        if (dir == "left"):
            # Shrink the left block by looking
            # it up from the history. Grow the right
            # in the usual way (with the add_spin method on the left).
            self.block1 = self.history["left"][self.block1.size-1]
            self.block2.add_spin(self.new_ops, "left")
        else:
            # Otherwise, do the exact opposite.
            self.block1.add_spin(self.new_ops, "right")
            self.block2 = self.history["right"][self.block2.size-1]
        # Make the Hamiltonian and get the ground state.
        hamiltonian_sb = self.make_superblock_hamiltonian()
        w, v = eigsh(hamiltonian_sb, k=1)
        energy = w[0]
        psi0 = v[:, 0]
        # Truncate the operator that got larger.
        trans_op1, trans_op2 = self.make_truncation_operators(psi0)
        if (dir == "left"):
            # Truncate the block on the right.
            self.block2.truncate_operators(trans_op2)
        else:
            # Truncate the block on the left.
            self.block1.truncate_operators(trans_op1)
        # Return the ground state and energy.
        return (energy, psi0)
