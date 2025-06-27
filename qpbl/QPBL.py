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

    def __init__(self, theta: np.ndarray , wire = None, IndexQubits: int = 2, LQubits: int = 5, QQubits: int = 2, O : str = "X"):
        self._hyperparameters = {
            "IndexQubits": IndexQubits,
            "LQubits": LQubits,
            "QQubits": QQubits,
            "O": O ,

        }
        super().__init__(theta,wires=wire)
    
    @staticmethod
    def compute_decomposition(theta: np.ndarray, wires, IndexQubits: int, LQubits: int = 5, QQubits: int = 2, O : str = "X") -> list[Operation]:
        """Compute the decomposition of the Eigen operation."""
        n_params = len(theta)
        offset_L = n_params + IndexQubits
        offset_U = n_params + IndexQubits + LQubits
        op_list = []
        for i in range(LQubits):
            op_list.append(qml.Hadamard(wires=i+offset_L))
        for i in range(LQubits):
            if O == 'XX':
                op_list.append(
                    qml.ctrl(
                        (
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
                            @(qml.PauliX(wires=offset_U)@qml.PauliX(wires=i+offset_U+1))
                        )**(2**i), control=offset_U-1-i
                    )
                )
            elif O == 'YY':
                op_list.append(
                    qml.ctrl(
                        (
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
                            @(qml.PauliY(wires=offset_U)@qml.PauliY(wires=offset_U+1))
                        )**(2**i), control=offset_U-1-i
                    )
                )
            elif O == 'YZ':
                op_list.append(
                    qml.ctrl(
                        (
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
                            @(qml.PauliZ(wires=offset_U)@qml.PauliZ(wires=offset_U+1))
                        )**(2**i), control=offset_U-1-i
                    )
                )
            elif O == 'II':
                op_list.append(
                    qml.ctrl(
                        (
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits)@
                            qml.ctrl(qml.Z(offset_U)@qml.Z(offset_U+1), control=IndexQubits+1)
                            @(qml.I(wires=offset_U)@qml.I(wires=offset_U+1))
                        )**(2**i), control=offset_U-1-i
                    )
                )
        op_list.append(qml.adjoint(qml.QFT(wires=range(offset_L, offset_U))))
        return op_list
    

class Utils():

    def __init__(self):
        pass

    @staticmethod
    def int_to_bin(n, num_bits)-> tuple:
        return tuple(int(x) for x in format(n, f'0{num_bits}b'))