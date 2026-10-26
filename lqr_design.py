"""Discrete LQR design with explicit stabilizability / detectability checks.

Written for a general (possibly lifted) model z+ = A z + B v, so the
Koopman models reuse exactly the same routine.
"""
import numpy as np
from scipy.linalg import solve_discrete_are, sqrtm


def pbh_stabilizable(A, B, tol=1e-8):
    """(A,B) stabilizable  <=>  rank[lam I - A, B] = n  for every |lam| >= 1."""
    n = A.shape[0]
    bad = []
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1.0 - 1e-9:
            M = np.hstack([lam * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M, tol=tol * max(1.0, np.linalg.norm(M))) < n:
                bad.append(lam)
    return len(bad) == 0, bad


def pbh_detectable(A, Cq, tol=1e-8):
    """(A, Cq) detectable  <=>  rank[lam I - A; Cq] = n  for every |lam| >= 1.
    For LQR use Cq = Q^{1/2}: every mode NOT penalized by Q must be stable."""
    ok, bad = pbh_stabilizable(A.T, Cq.T, tol)
    return ok, bad


def dlqr(A, B, Q, R, check=True):
    """Gain K for v = -K z minimising sum z'Qz + v'Rv  subject to z+ = Az + Bv.

    Returns K, P (Riccati solution) and the closed-loop eigenvalues."""
    if check:
        ok_s, bad_s = pbh_stabilizable(A, B)
        Qh = np.real(sqrtm(Q))
        ok_d, bad_d = pbh_detectable(A, Qh)
        if not ok_s:
            raise ValueError(f"(A,B) not stabilizable; bad modes: {bad_s}")
        if not ok_d:
            raise ValueError(f"(A,Q^1/2) not detectable; bad modes: {bad_d}")
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eig_cl = np.linalg.eigvals(A - B @ K)
    if np.max(np.abs(eig_cl)) >= 1.0:
        raise RuntimeError(f"closed loop unstable: max|eig| = {np.max(np.abs(eig_cl))}")
    return K, P, eig_cl