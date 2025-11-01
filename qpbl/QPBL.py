r""""
Quantum Phase-Based Learning (QPBL) module
This module implements the Eigen and Test operations used in Quantum Phase-Based Learning (QPBL).
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operation, AnyWires


class Eigen(Operation):
    num_wires = AnyWires
    #num_params = 6

    def __init__(self, theta: np.ndarray , wire = None, IndexQubits: int = 2, LQubits: int = 5, QQubits: int = 2, O : np.ndarray = None, T : str = "X"):
        self._hyperparameters = {
            "IndexQubits": IndexQubits,
            "LQubits": LQubits,
            "QQubits": QQubits,
            "O": O ,
            "T": T,
        }
        super().__init__(theta,wires=wire)
    
    @staticmethod
    def compute_decomposition(theta: np.ndarray, wires, IndexQubits: int, LQubits: int = 5, QQubits: int = 2, O : np.ndarray = None, T : str = "X") -> list[Operation]:
        """Compute the decomposition of the Eigen operation."""
        n_params = 0 #len(theta) Temporaly there is not going to be training by
        offset_L = n_params + IndexQubits
        offset_U = n_params + IndexQubits + LQubits
        n_qubits = n_params + IndexQubits + LQubits + QQubits
        op_list = []
        for i in range(LQubits):
            op_list.append(qml.Hadamard(wires=i+offset_L))
        for i in range(LQubits):
            #if O is not None:
            op_list.append(qml.ctrl(qml.QubitUnitary(Utils.powm(O,i), wires=range(offset_U,n_qubits)), control=offset_U-1-i))
            # else: # Use templates (Be careful since templates are defined for 2 qubits only)
            #     if O == 'XX':
            #         op_list.append(
            #             qml.ctrl(
            #                 (
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
            #                     @(qml.PauliX(wires=offset_U)@qml.PauliX(wires=offset_U+1))
            #                 )**(2**i), control=offset_U-1-i
            #             )
            #         )
            #     elif O == 'YY':
            #         op_list.append(
            #             qml.ctrl(
            #                 (
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
            #                     @(qml.PauliY(wires=offset_U)@qml.PauliY(wires=offset_U+1))
            #                 )**(2**i), control=offset_U-1-i
            #             )
            #         )
            #     elif O == 'YZ':
            #         op_list.append(
            #             qml.ctrl(
            #                 (
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
            #                     @(qml.PauliZ(wires=offset_U)@qml.PauliZ(wires=offset_U+1))
            #                 )**(2**i), control=offset_U-1-i
            #             )
            #         )
            #     elif O == 'II':
            #         op_list.append(
            #             qml.ctrl(
            #                 (
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
            #                     qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
            #                     @(qml.I(wires=offset_U)@qml.I(wires=offset_U+1))
            #                 )**(2**i), control=offset_U-1-i
            #             )
            #         )
        op_list.append(qml.adjoint(qml.QFT(wires=range(offset_L, offset_U))))
        
        return op_list

class Mottonen(Operation):
    num_wires = AnyWires
    
    def __init__(self, theta: np.ndarray , wire = None, QQubits: int =2, offset_U: int = 5):
        self._hyperparameters = {
            "QQubits": QQubits,
            "offset_U": offset_U,
        }
        super().__init__(theta,wires=wire)
    
    @staticmethod
    def compute_decomposition(theta: np.ndarray, wires, QQubits: int = 2,  offset_U: int = 5) -> list[Operation]:
        op_list = []
        c = 2**QQubits -2
        for k,i in enumerate(range(QQubits-1, -1, -1)):
            for j in range(2**k-1,-1,-1):
                if k>0:
                    op_list.append(qml.ctrl(qml.adjoint(qml.RY(theta[c], wires=k+offset_U)), control=range(offset_U,k+offset_U), control_values=Utils.int_to_bin(j,k)))
                else:
                    op_list.append(qml.adjoint(qml.RY(theta[c], wires=k+QQubits)))
                c = c - 1
        return op_list

class Utils():

    def __init__(self):
        pass

    @staticmethod
    def int_to_bin(n, num_bits)-> tuple:
        return tuple(int(x) for x in format(n, f'0{num_bits}b'))
    
    @staticmethod
    def powm(M,p):
        if p == 0:
            return M
        T = M@M
        for i in range(1,p):
            T = T@T
        return T