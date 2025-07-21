import numpy as np
import sympy as sp
from typing import Union


def invert_matrix(matrix: Union[np.ndarray, sp.Matrix],
                  symbolic: bool) -> Union[np.ndarray, sp.Matrix]:
    """
    Helper function that inverts matrices in both numeric and symbolic mode.
    This function raises an error message, if the matrix is not invertible.
    """
    if symbolic:
        if matrix.det() == 0:
            raise ValueError("Matrix is not invertible (symbolic)")
        return matrix.inv()
    else:
        if np.linalg.det(matrix) == 0:
            raise ValueError("Matrix is not invertible (numeric)")
        return np.linalg.inv(matrix)


def is_unitary(U: Union[np.ndarray, sp.Matrix],
               symbolic: bool = False,
               atol: float = 1e-10) -> bool:
    """
    This helper function checks, if a matrix is unitary.
    This needs to be verified for basis change operations.
    It operates in numeric mode. Hence, the matrix is first converted into a np.ndarray.
    """
    U_eval = np.array(U.evalf()).astype(
        np.complex128) if symbolic else np.array(U)
    return np.allclose(U_eval.conj().T @ U_eval, np.eye(U_eval.shape[0]), atol=atol)


def sanitize_vector(vec: Union[np.ndarray, list, tuple, sp.Matrix],
                    symbolic: bool) -> Union[np.ndarray, sp.Matrix]:
    """
    This helper function ensures that the vector (vec) is a type fitting 
    for the mode we are working in, i.e. symbolic or numeric.
    """
    return sp.Matrix(vec) if symbolic else np.array(vec, dtype=float)


def get_identity(size: int,
                 symbolic: bool) -> Union[np.ndarray, sp.Matrix]:
    """
    This helper function returns the identity matrix needed for the 
    current mode: symbolic or numeric.
    """
    return sp.eye(size) if symbolic else np.eye(size)


def print_symbolic_matrix(matrix: sp.Matrix,
                          name: str = "Matrix") -> None:
    """
    This helper function prints a symbolic matrix, which is 
    particularly helpful when debugging a symbolic Hamiltonian. 
    """
    print(f"\n{name}:")
    sp.pprint(matrix, use_unicode=True)
