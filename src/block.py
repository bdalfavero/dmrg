import numpy as np
import scipy.sparse as sp

class Block:
    # Block for the DMRG algorithm.
    # Consists of a line of spins.

    def __init__(self, hamiltonian, end_ops, consts):
        # hamtiltonian: initial Hamiltonian of block
        # end_ops: 1 * S^\sigma for \sigma \in {x, y, z}
        #         i.e. operators for the end of the chain.
        # consts: Coupling constants [J_x, J_y, J_z] for 2 spins.
        
        self.dim = hamiltonian.shape[0]
        self.size = self.dim.bit_length() - 1
        self.hamiltonian = hamiltonian
        self.end_ops = end_ops
        self.constants = consts

    def add_spin(self, new_ops, dir):
        """
        Add a spin to the block at the appropriate end.
        Parameters:
        new_ops: List of the new operators, e.g. [S^x, S^y, S^z].
        dir: String, either "left" or "right".
        """

        # Make an identity operator for the new spin space.
        eye1 = np.eye(new_ops[0].shape[0])
        # Make an identity for the old system size.
        eye_old = np.eye(self.hamiltonian.shape[0])
        # Add a term for J S_i-1 dot S_i at the end.
        if (dir == "right"):
            self.hamiltonian = np.kron(self.hamiltonian, eye1)
        else:
            self.hamiltonian = np.kron(eye1, self.hamiltonian)
        for i in range(len(new_ops)):
            if (dir == "right"):
                self.hamiltonian += self.constants[i] \
                                 * np.kron(self.end_ops[i], new_ops[i])
            else:
                self.hamiltonian += self.constants[i] \
                                 * np.kron(new_ops[i], self.end_ops[i])            
        # Update the end operators to I_l \otimes S^sigma.
        for i in range(len(self.end_ops)):
            if (dir == "right"):
                self.end_ops[i] = np.kron(eye_old, new_ops[i])
            else:
                self.end_ops[i] = np.kron(new_ops[i], eye_old)
        # Update the operator size and spin count.
        self.dim = self.hamiltonian.shape[0]
        self.size += 1

    def truncate_operators(self, trans_op):
        """
        Truncate the operators into a new basis.
        trans_op: matrix of new basis vectors
        """

        # Truncate the operators.
        self.hamiltonian = trans_op.conj().T @ self.hamiltonian @ trans_op
        for i in range(len(self.end_ops)):
            self.end_ops[i] = trans_op.conj().T @ self.end_ops[i] @ trans_op
        # Update the block dimension.
        self.dim = self.hamiltonian.shape[0]
