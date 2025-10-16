import numpy as np
import sympy as sp
from sympy.matrices.common import NonInvertibleMatrixError
from typing import Union, Optional
from pathlib import Path


def invert_matrix(matrix: Union[np.ndarray, sp.Matrix],
                  symbolic: bool) -> Union[np.ndarray, sp.Matrix]:
    """
    Compute the inverse of a matrix.

    Parameters:
    - matrix: numpy.ndarray or sympy.Matrix
    - symbolic: True for SymPy backend, False for NumPy

    Returns:
    - Inverse of matrix
    
    This function raises an error message, if the matrix is not invertible.
    """
    if symbolic:
        #if matrix.det() == 0: # ==0 not reliable, new solution needed
        #    raise ValueError("Matrix is not invertible (symbolic)")
        assert isinstance(matrix, sp.Matrix), f"Expected sympy.Matrix for symbolic inversion, got {type(matrix)}"
        try:
            return matrix.inv(method='LU') # LU method is more robust than default
        except NonInvertibleMatrixError:
            raise ValueError("Matrix is not invertible (symbolic) under current assumptions.")
    else:
        assert isinstance(matrix, np.ndarray), f"Expected np.ndarray for numeric inversion, got {type(matrix)}"
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not invertible (numerically)")

def hermitian_conjugate(matrix: Union[np.ndarray, sp.Matrix],
                        symbolic: bool) -> Union[np.ndarray, sp.Matrix]:
    """
    Compute the Hermitian conjugate (dagger) of a matrix.

    Parameters:
    - matrix: numpy.ndarray or sympy.Matrix
    - symbolic: True for SymPy backend, False for NumPy

    Returns:
    - Hermitian conjugate of matrix
    """
    if symbolic:
        return matrix.H
    else:
        return matrix.conj().T

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

def is_scalar(x) -> bool:
    '''
    Tiny helper to identify scalar-like values (int, float, np.number, sp.Basic).
    '''
    return (isinstance(x, (int, float, np.number, sp.Basic)))

def sanitize_vector(
    veclike: Union[np.ndarray, list, tuple, sp.Matrix, int, float, sp.Basic],
    symbolic: bool,
    expected_dim: Optional[int] = None,
    ) -> Union[np.ndarray, sp.Matrix]:
    """
    Coerce `veclike` into a vector with consistent backend types.

    - numeric: returns np.ndarray with shape (n,)
    - symbolic: returns sp.Matrix with shape (n, 1)

    Policy:
    - If expected_dim == 1 and input is scalar-like, wrap to length-1 vector.
    - If expected_dim > 1 and input is scalar-like, raise ValueError.
    - Otherwise coerce and validate shape == expected_dim (when provided).
    """
    if isinstance(veclike, np.ndarray) and veclike.ndim == 0:
        veclike = veclike.item() # unwrap 0-dim ndarray to scalar

    # Dim-conditional scalar handling
    if expected_dim is not None and is_scalar(veclike):
        if expected_dim == 1:
            veclike = [veclike]             # wrap scalar for 1D
        else:
            raise ValueError(f"Expected vector of length {expected_dim}, got scalar")

    # Coerce to backend type
    vec = sp.Matrix(veclike) if symbolic else np.asarray(veclike, dtype=float)

    # Normalize shapes
    if symbolic:
        # ensure column vector
        if vec.cols != 1:
            vec = sp.Matrix(vec).reshape(vec.rows * vec.cols, 1)
    else:
        vec = np.ravel(vec)  # 1-D view

    # Enforce expected length
    if expected_dim is not None:
        n = vec.shape[0] # length of vector
        if n != expected_dim:
            raise ValueError(f"Expected vector of length {expected_dim}, got {n}")

    return vec

def sanitize_matrix(
    matlike: Union[np.ndarray, list, tuple, sp.Matrix, int, float, sp.Basic],
    symbolic: bool,
    expected_size: int
    )-> Union[np.ndarray, sp.Matrix]:

    if is_scalar(matlike):
        matlike = [matlike] # ensure indexable for 1D
    elif isinstance(matlike, np.ndarray) and matlike.ndim == 0:
        matlike = [matlike.item()] # ensure indexable for 1D

    if symbolic:
        mat = sp.Matrix(matlike)
    else:
        mat = np.asarray(matlike, dtype=complex)

    if mat.shape != (expected_size, expected_size):
        raise ValueError(
            f"H(k) must be {expected_size}x{expected_size}, got {mat.shape}.")
    return mat

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

def save_result(result, path, symbolic: bool):
    """
    Minimal saver:
      - symbolic=True  -> write <path>.tex and <path>.pretty.txt
      - symbolic=False -> write <path>.npy
    Returns a list of written file paths.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written = []
    if symbolic:
        if not isinstance(result, sp.MatrixBase):
            result = sp.Matrix(result)
        tex_path = path.with_suffix(".tex")
        pretty_path = path.with_suffix(".pretty.txt")
        tex_path.write_text(sp.latex(result))
        pretty_path.write_text(sp.pretty(result))
        written += [tex_path, pretty_path]
    else:
        arr = np.asarray(result)
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, arr)
        written.append(npy_path)
    return written