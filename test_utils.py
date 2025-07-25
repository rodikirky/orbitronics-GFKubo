import numpy as np
import sympy as sp
import pytest

from utils import (
    invert_matrix,
    is_unitary,
    sanitize_vector,
    get_identity,
    print_symbolic_matrix
)

# ---------- Tests for invert_matrix ----------


def test_invert_matrix_numeric():
    A = np.array([[2, 0], [0, 1]])
    A_inv = invert_matrix(A, symbolic=False)  # numeric mode
    assert np.allclose(A @ A_inv, np.eye(2))


def test_invert_matrix_symbolic():
    a, b = sp.symbols("a b")
    A = sp.Matrix([[a, 0], [0, b]])
    # invesion with symbols
    A_inv = invert_matrix(A, symbolic=True)  # symbolic mode
    assert A @ A_inv == sp.eye(2)
    # inversion with numbers substituted for symbols
    A_eval = A.subs({a: 2, b: 1})
    A_eval_inv = invert_matrix(A_eval, symbolic=True)  # still in symbolic mode
    assert np.allclose(
        np.array(A_eval @ A_eval_inv).astype(np.float64), np.eye(2))


def test_invert_matrix_raises_on_singular_numeric():
    A = np.array([[1, 2], [2, 4]])  # Singular matrix
    with pytest.raises(ValueError, match="invertible"):  # ought to raise "ValueError"
        invert_matrix(A, symbolic=False)


def test_invert_matrix_raises_on_singular_symbolic():
    A = sp.Matrix([[1, 2], [2, 4]])  # Singular symbolic
    with pytest.raises(ValueError, match="invertible"):  # ought to raise "ValueError"
        invert_matrix(A, symbolic=True)

# ---------- Tests for is_unitary ----------


def test_is_unitary_true_numeric():
    U = np.array([[1, 0], [0, 1]])  # Identity is unitary
    assert is_unitary(U, symbolic=False)


def test_is_unitary_true_numeric_complex():
    # complex unitary matrix in numeric mode
    sqrt2 = np.sqrt(2)
    U_0 = np.array([[-1j/sqrt2, 0, 1j/sqrt2],
                   [1/sqrt2, 0, 1/sqrt2], [0, 1j, 0]])
    assert is_unitary(U_0, symbolic=False)


def test_is_unitary_false_numeric():
    M = np.array([[1, 2], [3, 4]])
    assert not is_unitary(M, symbolic=False)


def test_is_unitary_true_symbolic():
    U = sp.eye(2)
    assert is_unitary(U, symbolic=True)


def test_is_unitary_true_numeric_complex():
    # complex unitary matrix in symbolic mode
    sqrt2 = sp.sqrt(2)  # it is crucial to use sympy's sqrt here
    U_0 = sp.Matrix([[-sp.I/sqrt2, 0, sp.I/sqrt2], [1/sqrt2, 0,
                    1/sqrt2], [0, sp.I, 0]])  # complex unitary matrix
    assert is_unitary(U_0, symbolic=True)


def test_is_unitary_false_symbolic():
    M = sp.Matrix([[1, 2], [3, 4]])
    assert not is_unitary(M, symbolic=True)

# ---------- Tests for sanitize_vector ----------


def test_sanitize_vector_numeric():
    v = [1, 2, 3]
    result = sanitize_vector(v, symbolic=False)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.array([1, 2, 3]))


def test_sanitize_vector_symbolic():
    v = [1, 2, 3]
    result = sanitize_vector(v, symbolic=True)
    assert isinstance(result, sp.Matrix)
    assert result.shape == (3, 1)

# ---------- Tests for get_identity ----------


def test_get_identity_numeric():
    I = get_identity(3, symbolic=False)
    assert isinstance(I, np.ndarray)
    assert I.shape == (3, 3)
    assert np.allclose(I, np.eye(3))


def test_get_identity_symbolic():
    I = get_identity(3, symbolic=True)
    assert isinstance(I, sp.Matrix)
    assert I.shape == (3, 3)
    assert I == sp.eye(3)

# ---------- Test for print_symbolic_matrix (optional) ----------


def test_print_symbolic_matrix(capsys):
    A = sp.Matrix([[1, 0], [0, 1]])
    print_symbolic_matrix(A, name="TestMatrix")
    captured = capsys.readouterr()
    assert "TestMatrix" in captured.out  # Check that the name label was printed
    # check that the matrix contents include a "1" (just a sanity check)
    assert "1" in captured.out
