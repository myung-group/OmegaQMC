import jax
import jax.numpy as jnp
import numpy as np
# from functools import partial
from OmegaQMC.shell import (
    read_shell, evaluate_cusp_s, evaluate_cusp_s_vgl,
)
from OmegaQMC.utils import laplacian_linearize
# from OmegaQMC.constants import EE_CUSP_VALUE
# JASTROW_EE_L_CUT, JASTROW_EE_M_POWER


# --- B-spline helpers (QMCPACK BsplineFunctor convention) ---

def _build_bspline_coefs(params, delta_r, cusp_val):
    """Map N user params → (N+4) full B-spline coefficients.

    Following QMCPACK's BsplineFunctor.h:
      coefs[1] = P[0], coefs[2] = P[1], ..., coefs[N] = P[N-1]
      coefs[0] = coefs[2] - 2 * delta_r * cusp_val   (cusp)
      coefs[N+1] = coefs[N+2] = coefs[N+3] = 0       (boundary)
    """
    # n = params.shape[0]
    c0 = params[1] - 2.0 * delta_r * cusp_val
    coefs = jnp.concatenate([
        c0[jnp.newaxis],
        params,
        jnp.zeros(3),
    ])
    return coefs


def _bspline_eval(r, coefs, delta_r_inv, max_index):
    """Evaluate cubic B-spline at distances r.

    Uses the uniform-knot cubic basis from QMCPACK.
    r is an array of distances; returns u(r) for each.
    """
    u = r * delta_r_inv
    i = jnp.clip(
        jnp.floor(u).astype(jnp.int32), 0, max_index
    )
    t = u - i

    # Cubic B-spline basis values (uniform knots)
    tp = jnp.stack([t * t * t, t * t, t, jnp.ones_like(t)])
    basis = jnp.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0],
    ], dtype=r.dtype) / 6.0

    w = basis.T @ tp                      # (4, len(r))

    c0 = coefs[i]
    c1 = coefs[i + 1]
    c2 = coefs[i + 2]
    c3 = coefs[i + 3]

    return w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3


def _bspline_eval_vgl(r, coefs, delta_r_inv, max_index):
    """Evaluate cubic B-spline value, first and second derivative.

    Returns ``(u(r), u'(r), u''(r))`` — derivatives are with
    respect to the scalar distance ``r``.  Uses the same uniform-
    knot cubic basis as :func:`_bspline_eval` and shares the
    ``i / t / coefs[i+k]`` gather, so the work overlaps with the
    value-only path.

    Coefficient layout from :func:`_build_bspline_coefs` enforces
    ``coefs[N+1] = coefs[N+2] = coefs[N+3] = 0``, which makes
    the evaluated ``(u, u', u'')`` go to zero at ``r = r_cut``;
    callers are still expected to apply a ``cutoff_mask = (r <
    r_cut)`` to clip the polynomial extrapolation past the
    cutoff.
    """
    u = r * delta_r_inv
    i = jnp.clip(
        jnp.floor(u).astype(jnp.int32), 0, max_index
    )
    t = u - i
    one = jnp.ones_like(t)
    zero = jnp.zeros_like(t)

    tp_v = jnp.stack([t * t * t,         t * t,    t,    one])
    tp_d1 = jnp.stack([3.0 * t * t,      2.0 * t,  one,  zero])
    tp_d2 = jnp.stack([6.0 * t,          2.0 * one, zero, zero])
    basis = jnp.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0],
    ], dtype=r.dtype) / 6.0

    w = basis.T @ tp_v                    # (4, len(r))
    wd1 = basis.T @ tp_d1
    wd2 = basis.T @ tp_d2

    c0 = coefs[i]
    c1 = coefs[i + 1]
    c2 = coefs[i + 2]
    c3 = coefs[i + 3]

    val = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3
    d1 = (
        wd1[0] * c0 + wd1[1] * c1
        + wd1[2] * c2 + wd1[3] * c3
    ) * delta_r_inv
    d2 = (
        wd2[0] * c0 + wd2[1] * c1
        + wd2[2] * c2 + wd2[3] * c3
    ) * (delta_r_inv * delta_r_inv)
    return val, d1, d2


def _compute_eeI_num_params(N_eI, N_ee):
    """Number of free eeI polynomial parameters."""
    NumGamma = (
        (N_eI + 1) * (N_eI + 2) // 2 * (N_ee + 1)
    )
    NumConstraints = (
        (2 * N_eI + 1) + (N_eI + N_ee + 1)
    )
    return NumGamma - NumConstraints


def _sanitize_J3_eeI_params(params_j3, N_eI=3,
                             N_ee=3):
    """Validate and sanitize J3_eeI parameter dict.

    Each value must be a 1-D array of length
    ``_compute_eeI_num_params(N_eI, N_ee)``.
    Invalid or wrong-size entries are replaced with
    zero arrays and a warning is emitted.

    Parameters
    ----------
    params_j3 : dict
        The ``params_corr["J3_eeI"]`` sub-dict.
    N_eI, N_ee : int
        Polynomial orders (must match the config
        used when the driver was constructed).

    Returns
    -------
    dict
        Sanitized copy of *params_j3*.
    """
    import warnings
    num_params = _compute_eeI_num_params(N_eI, N_ee)
    out = {}
    for k, v in params_j3.items():
        try:
            arr = jnp.asarray(v, dtype=jnp.float64)
        except Exception:
            warnings.warn(
                f"J3_eeI['{k}']: cannot convert "
                f"to array; replacing with zeros"
            )
            out[k] = jnp.zeros(num_params)
            continue
        if arr.ndim != 1:
            warnings.warn(
                f"J3_eeI['{k}']: expected 1-D "
                f"array, got ndim={arr.ndim}; "
                f"replacing with zeros"
            )
            out[k] = jnp.zeros(num_params)
        elif arr.shape[0] != num_params:
            warnings.warn(
                f"J3_eeI['{k}']: expected "
                f"{num_params} params (for "
                f"N_eI={N_eI}, N_ee={N_ee}), "
                f"got {arr.shape[0]}; "
                f"replacing with zeros"
            )
            out[k] = jnp.zeros(num_params)
        else:
            out[k] = arr
    return out


# --- eeI (three-body) Jastrow helpers ---
#
# Port of QMCPACK's PolynomialFunctor3D.  The functional
# form is:
#   u(r12,r1I,r2I) = [(r1I-L)(r2I-L)]^C
#                    * sum_{l,m,n} gamma(l,m,n)
#                      * r1I^l * r2I^m * r12^n
# with L = r_cut/2, C = 3 (continuity order), and
# gamma symmetric in (l,m).  Zero-cusp constraints
# reduce the free parameter count.

def _build_eeI_constraint_map(N_eI, N_ee, r_cut):
    """Build the linear map from free params to gammas.

    Ports the constraint logic from QMCPACK
    ``PolynomialFunctor3D::resize`` and ``reset_gamma``.

    Parameters
    ----------
    N_eI : int
        Max polynomial order in electron-ion distances.
    N_ee : int
        Max polynomial order in electron-electron
        distance.
    r_cut : float
        Cutoff radius in bohr.

    Returns
    -------
    A : jnp.ndarray, shape (NumGamma, NumParams)
        Matrix mapping free parameters to the full
        gamma vector: ``gamma_vec = A @ free_params``.
    ls, ms, ns : jnp.ndarray
        Index arrays for the unique gamma entries
        (l >= m), used by ``_vec_to_gamma_3d``.
    """
    import numpy as np
    L = 0.5 * r_cut
    C = 3  # continuity order

    # --- Index map (unique gammas with l >= m) ---
    NumGamma = (
        (N_eI + 1) * (N_eI + 2) // 2 * (N_ee + 1)
    )
    index = np.zeros(
        (N_eI + 1, N_eI + 1, N_ee + 1), dtype=int
    )
    num = 0
    idx_l, idx_m, idx_n = [], [], []
    for m in range(N_eI + 1):
        for l in range(m, N_eI + 1):
            for n in range(N_ee + 1):
                index[l, m, n] = num
                index[m, l, n] = num
                idx_l.append(l)
                idx_m.append(m)
                idx_n.append(n)
                num += 1
    assert num == NumGamma

    # --- Constraint matrix ---
    NumConstraints = (2 * N_eI + 1) + (N_eI + N_ee + 1)
    C_mat = np.zeros((NumConstraints, NumGamma))

    # e-e no-cusp constraints (n=1 coefficients)
    for k in range(2 * N_eI + 1):
        for m_idx in range(k + 1):
            l_idx = k - m_idx
            if l_idx <= N_eI and m_idx <= N_eI:
                i = index[l_idx, m_idx, 1]
                if l_idx > m_idx:
                    C_mat[k, i] = 2.0
                elif l_idx == m_idx:
                    C_mat[k, i] = 1.0

    # e-I no-cusp constraints
    row_offset = 2 * N_eI + 1
    for kp in range(N_eI + N_ee + 1):
        if kp <= N_ee:
            C_mat[
                row_offset + kp,
                index[0, 0, kp]
            ] = float(C)
            if N_eI >= 1:
                C_mat[
                    row_offset + kp,
                    index[0, 1, kp]
                ] = -L
        for l_idx in range(1, kp + 1):
            n_idx = kp - l_idx
            if (n_idx >= 0 and n_idx <= N_ee
                    and l_idx <= N_eI):
                C_mat[
                    row_offset + kp,
                    index[l_idx, 0, n_idx]
                ] = float(C)
                if N_eI >= 1:
                    C_mat[
                        row_offset + kp,
                        index[l_idx, 1, n_idx]
                    ] = -L

    # --- Row reduction with partial pivoting ---
    IndepVar = np.zeros(NumGamma, dtype=bool)
    col = -1
    for row in range(NumConstraints):
        while True:
            col += 1
            if col >= NumGamma:
                break
            max_loc = row
            max_abs = abs(C_mat[row, col])
            for ri in range(row + 1, NumConstraints):
                av = abs(C_mat[ri, col])
                if av > max_abs:
                    max_loc = ri
                    max_abs = av
            if max_abs < 1e-6:
                IndepVar[col] = True
                continue
            break
        if col >= NumGamma:
            break
        C_mat[[row, max_loc]] = C_mat[[max_loc, row]]
        C_mat[row] /= C_mat[row, col]
        for ri in range(NumConstraints):
            if ri != row:
                C_mat[ri] -= (
                    C_mat[ri, col] * C_mat[row]
                )
    for c in range(col + 1, NumGamma):
        IndepVar[c] = True

    NumParams = int(np.sum(IndepVar))
    assert NumParams == NumGamma - NumConstraints

    # --- Build A matrix: gamma_vec = A @ free_params ---
    A = np.zeros((NumGamma, NumParams))
    indep_cols = np.where(IndepVar)[0]
    dep_cols = np.where(~IndepVar)[0]

    for p, j in enumerate(indep_cols):
        A[j, p] = 1.0

    for c, i in enumerate(dep_cols):
        for p, j in enumerate(indep_cols):
            A[i, p] = -C_mat[c, j]

    ls = jnp.array(idx_l, dtype=jnp.int32)
    ms = jnp.array(idx_m, dtype=jnp.int32)
    ns = jnp.array(idx_n, dtype=jnp.int32)

    return jnp.array(A), ls, ms, ns


def _vec_to_gamma_3d(gamma_vec, ls, ms, ns, shape):
    """Convert flat gamma vector to symmetric 3-D array.

    Parameters
    ----------
    gamma_vec : jnp.ndarray, shape (NumGamma,)
    ls, ms, ns : index arrays from constraint map
    shape : tuple (N_eI+1, N_eI+1, N_ee+1)
    """
    g = jnp.zeros(shape, dtype=gamma_vec.dtype)
    g = g.at[ls, ms, ns].set(gamma_vec)
    g = g.at[ms, ls, ns].set(gamma_vec)
    return g


def _power_table(r, max_pow):
    """Build [r^0, r^1, ..., r^max_pow] for each r.

    Returns shape ``(max_pow + 1, len(r))``.
    """
    pows = [jnp.ones_like(r)]
    for _ in range(max_pow):
        pows.append(pows[-1] * r)
    return jnp.stack(pows)


def _eval_eeI_poly(r_12, r_1I, r_2I,
                   gamma_3d, L, N_eI, N_ee):
    """Evaluate the eeI polynomial on a batch of triplets.

    Parameters
    ----------
    r_12, r_1I, r_2I : shape (T,)
    gamma_3d : shape (N_eI+1, N_eI+1, N_ee+1)
    L : half-cutoff
    N_eI, N_ee : polynomial orders
    """
    p1 = _power_table(r_1I, N_eI)
    p2 = _power_table(r_2I, N_eI)
    p12 = _power_table(r_12, N_ee)
    P = jnp.einsum(
        'lmn,lt,mt,nt->t', gamma_3d, p1, p2, p12
    )
    envelope = ((r_1I - L) * (r_2I - L)) ** 3
    return P * envelope


def _eval_eeI_poly_vgl(r_12, r_1I, r_2I,
                       gamma_3d, L, N_eI, N_ee):
    """Evaluate ``u = P · E`` and the nine derivatives of ``u``.

    Returns ``(u, u_d1I, u_d2I, u_d12, u_d1I_1I, u_d2I_2I,
    u_d12_12, u_d1I_12, u_d2I_12)`` — each shape ``(T,)``.

    The mixed ``∂²u/(∂r_1I ∂r_2I)`` is *not* returned: it does
    not appear in either electron's per-electron Laplacian since
    ``r_1I`` depends only on electron 1 and ``r_2I`` only on
    electron 2.

    Parameters
    ----------
    r_12, r_1I, r_2I : shape (T,)
    gamma_3d : shape (N_eI+1, N_eI+1, N_ee+1)
    L : half-cutoff (envelope vanishes at ``r_1I, r_2I = L``)
    N_eI, N_ee : polynomial orders
    """
    p1 = _power_table(r_1I, N_eI)
    p2 = _power_table(r_2I, N_eI)
    p12 = _power_table(r_12, N_ee)

    # Shifted-index power tables for analytical polynomial
    # derivatives:
    #   pd1[k] = k · r^{k-1}        (k ≥ 1, else 0)
    #   pd2[k] = k(k-1) · r^{k-2}   (k ≥ 2, else 0)
    ks_eI = jnp.arange(N_eI + 1, dtype=p1.dtype)
    ks_ee = jnp.arange(N_ee + 1, dtype=p12.dtype)
    z1 = jnp.zeros_like(p1[:1])
    z2 = jnp.zeros_like(p12[:1])
    p1_s1 = jnp.concatenate([z1, p1[:-1]], axis=0)
    p2_s1 = jnp.concatenate([z1, p2[:-1]], axis=0)
    p12_s1 = jnp.concatenate([z2, p12[:-1]], axis=0)
    p1_s2 = jnp.concatenate([z1, z1, p1[:-2]], axis=0)
    p2_s2 = jnp.concatenate([z1, z1, p2[:-2]], axis=0)
    p12_s2 = jnp.concatenate([z2, z2, p12[:-2]], axis=0)
    pd1_a = ks_eI[:, None] * p1_s1
    pd1_b = ks_eI[:, None] * p2_s1
    pd1_c = ks_ee[:, None] * p12_s1
    pd2_a = (ks_eI * (ks_eI - 1.0))[:, None] * p1_s2
    pd2_b = (ks_eI * (ks_eI - 1.0))[:, None] * p2_s2
    pd2_c = (ks_ee * (ks_ee - 1.0))[:, None] * p12_s2

    # Polynomial value and its nine derivatives.
    es = 'lmn,lt,mt,nt->t'
    P_val = jnp.einsum(es, gamma_3d, p1, p2, p12)
    P_d1I = jnp.einsum(es, gamma_3d, pd1_a, p2, p12)
    P_d2I = jnp.einsum(es, gamma_3d, p1, pd1_b, p12)
    P_d12 = jnp.einsum(es, gamma_3d, p1, p2, pd1_c)
    P_d1I_1I = jnp.einsum(es, gamma_3d, pd2_a, p2, p12)
    P_d2I_2I = jnp.einsum(es, gamma_3d, p1, pd2_b, p12)
    P_d12_12 = jnp.einsum(es, gamma_3d, p1, p2, pd2_c)
    P_d1I_12 = jnp.einsum(es, gamma_3d, pd1_a, p2, pd1_c)
    P_d2I_12 = jnp.einsum(es, gamma_3d, p1, pd1_b, pd1_c)

    # Envelope (s_1 s_2)^3 and its derivatives.  Only the
    # five derivatives that actually appear in the per-
    # electron Laplacian are needed.
    s1 = r_1I - L
    s2 = r_2I - L
    s1_2 = s1 * s1
    s1_3 = s1_2 * s1
    s2_2 = s2 * s2
    s2_3 = s2_2 * s2
    E = s1_3 * s2_3
    E_d1I = 3.0 * s1_2 * s2_3
    E_d2I = 3.0 * s1_3 * s2_2
    E_d1I_1I = 6.0 * s1 * s2_3
    E_d2I_2I = 6.0 * s1_3 * s2

    # u = P · E and its nine derivatives via Leibniz.
    u = P_val * E
    u_d1I = P_d1I * E + P_val * E_d1I
    u_d2I = P_d2I * E + P_val * E_d2I
    u_d12 = P_d12 * E
    u_d1I_1I = (
        P_d1I_1I * E
        + 2.0 * P_d1I * E_d1I
        + P_val * E_d1I_1I
    )
    u_d2I_2I = (
        P_d2I_2I * E
        + 2.0 * P_d2I * E_d2I
        + P_val * E_d2I_2I
    )
    u_d12_12 = P_d12_12 * E
    u_d1I_12 = P_d1I_12 * E + P_d12 * E_d1I
    u_d2I_12 = P_d2I_12 * E + P_d12 * E_d2I

    return (u, u_d1I, u_d2I, u_d12,
            u_d1I_1I, u_d2I_2I, u_d12_12,
            u_d1I_12, u_d2I_12)


def _angular_cartesian(am, dr, rad_s):
    """Return GTO angular part for angular momentum am.

    am is a Python int (unrolled at JAX trace time).
    """
    if am == 0:
        return rad_s
    elif am == 1:
        return rad_s * dr
    elif am == 2:
        cd1 = jnp.sqrt(3.0)
        cd2 = cd1 * 0.5
        x, y, z = dr
        return rad_s * jnp.array([
            cd1*x*y,
            cd1*y*z,
            0.5*(2.0*z*z - x*x - y*y),
            cd1*x*z,
            cd2*(x*x - y*y)
        ])
    elif am == 3:
        cf1 = jnp.sqrt(2.5)*0.5
        cf2 = 3.0*cf1
        cf3 = jnp.sqrt(15.0)
        cf4 = jnp.sqrt(1.5)*0.5
        cf5 = jnp.sqrt(6.0)
        cf6 = 1.5
        cf7 = cf3*0.5
        x, y, z = dr
        return rad_s * jnp.array([
            y*(cf2*x*x - cf1*y*y),
            cf3*x*y*z,
            y*(cf5*z*z-cf4*(x*x+y*y)),
            z*(z*z - cf6*(x*x+y*y)),
            x*(cf5*z*z - cf4*(x*x+y*y)),
            z*cf7*(x*x-y*y),
            -x*(cf2*y*y-cf1*x*x)
        ])
    elif am == 4:
        cg1 = 2.9580398915498085
        cg2 = 6.2749501990055672
        cg3 = 2.0916500663351894
        cg4 = 1.1180339887498949
        cg5 = 6.7082039324993694
        cg6 = 2.3717082451262845
        cg7 = 3.1622776601683795
        cg8 = 0.55901699437494745
        cg9 = 3.3541019662496847
        cg10 = 0.73950997288745213
        cg11 = 4.4370598373247132
        x, y, z = dr
        return rad_s * jnp.array([
            cg1*(x*x*x*y-x*y*y*y),
            y*z*(cg2*x*x - cg3*y*y),
            x*y*cg4*(-x*x - y*y) + cg5*x*y*z*z,
            -cg6*x*x*y*z - cg6*y*y*y*z + cg7*y*z*z*z,
            (0.375*(x*x*x*x + y*y*y*y + 2.0*x*x*y*y) +
             z*z*z*z - 3.0*z*z*(x*x + y*y)),
            -(cg6*x*x*x*z + cg6*x*y*y*z - cg7*x*z*z*z),
            cg8*(y*y*y*y - x*x*x*x) + cg9*z*z*(x*x - y*y),
            -x*z*(cg2*y*y - cg3*x*x),
            cg10*(x*x*x*x + y*y*y*y) - cg11*x*x*y*y
        ])
    else:
        raise ValueError("shell.am > 4 is not supported yet.")


def _angular_cartesian_poly(am, dr):
    """Harmonic polynomial ``Y(dr)`` of a GTO shell.

    Same as :func:`_angular_cartesian` with ``rad_s`` set to
    ``1.0``, but always returns a 1-D array of shape
    ``(2 am + 1,)`` — even for ``am = 0`` — so the shape is
    uniform across shells.  ``am`` is a Python int unrolled
    at JAX trace time.
    """
    if am == 0:
        return jnp.ones(1)
    return _angular_cartesian(am, dr, 1.0)


def _angular_cartesian_vgl(am, dr):
    """Value and spatial gradient of the harmonic polynomial.

    Returns ``(Y, grad_Y)`` where ``Y`` has shape
    ``(2 am + 1,)`` and ``grad_Y`` has shape
    ``(2 am + 1, 3)``.  The Laplacian is zero by construction
    (solid harmonics are harmonic), so only ``(value, grad)``
    are returned.

    ``jax.jacobian`` unrolls at trace time because the
    dispatch in :func:`_angular_cartesian_poly` is a Python
    ``if`` on the constant ``am``.  For a small ``(2 am + 1,
    3)`` output this costs a negligible amount of memory.
    """
    Y = _angular_cartesian_poly(am, dr)
    if am == 0:
        return Y, jnp.zeros((1, 3))
    grad_Y = jax.jacobian(
        lambda d: _angular_cartesian_poly(am, d)
    )(dr)
    return Y, grad_Y


def _slater_det_assemble(mo_val, mo_grad, mo_lap):
    """Analytical (log|det|, ∇_e, ∇²_e) for a Slater det.

    Inputs:
      mo_val  — (n_e, n_mo)     φ_m(r_e)
      mo_grad — (n_e, n_mo, 3)  ∂φ_m/∂r_e
      mo_lap  — (n_e, n_mo)     ∇²_e φ_m(r_e)

    Alpha electrons are at even indices (``0, 2, 4, ...``)
    and beta at odd indices, matching the convention of
    :func:`log_slater_determinant`.

    Returns ``(log_val, grad_e, lap_e)`` with shapes
    ``()``, ``(n_e, 3)``, ``(n_e,)`` respectively.  The
    Slater identity used is the standard one:

        Γ[i, x]     = Σ_m (S⁻¹)[m, i] · G[i, m, x]
        ∇_i  log|S| = Γ[i, :]
        ∇²_i log|S| = Σ_m (S⁻¹)[m, i] · L[i, m] − |Γ[i]|²

    per spin block, then interleaved back to electron order.
    """
    def _block(S, G, L):
        A = jnp.linalg.inv(S)
        Gamma = jnp.einsum('ji,ijx->ix', A, G)
        lap = (
            jnp.einsum('ji,ij->i', A, L)
            - jnp.sum(Gamma * Gamma, axis=-1)
        )
        _, logdet = jnp.linalg.slogdet(S)
        return logdet, Gamma, lap

    S_a = mo_val[::2, :]
    S_b = mo_val[1::2, :]
    G_a = mo_grad[::2, :, :]
    G_b = mo_grad[1::2, :, :]
    L_a = mo_lap[::2, :]
    L_b = mo_lap[1::2, :]

    logdet_a, Gamma_a, lap_a = _block(S_a, G_a, L_a)
    logdet_b, Gamma_b, lap_b = _block(S_b, G_b, L_b)

    log_val = logdet_a + logdet_b
    n_e = mo_val.shape[0]
    grad_e = jnp.zeros((n_e, 3), dtype=mo_val.dtype)
    grad_e = grad_e.at[::2].set(Gamma_a)
    grad_e = grad_e.at[1::2].set(Gamma_b)
    lap_e = jnp.zeros(n_e, dtype=mo_val.dtype)
    lap_e = lap_e.at[::2].set(lap_a)
    lap_e = lap_e.at[1::2].set(lap_b)
    return log_val, grad_e, lap_e


def _radial_vgl(alpha, norm, r2, r):
    """Contracted Gaussian radial: value, ``R'/r``, and
    ``R'' + 2 R'/r``.

    For the contracted sum ``R(r) = Σ_p N_p · e^{-α_p r²}``,

        R(r)              = Σ_p N_p · e^{-α_p r²}
        R'(r)/r           = Σ_p N_p · (-2 α_p) · e^{-α_p r²}
        R''(r) + 2 R'(r)/r
                          = Σ_p N_p · (4 α_p² r² − 6 α_p)
                                · e^{-α_p r²}

    All three expressions are manifestly division-free, and
    are therefore safe at the electron-on-nucleus
    coincidence ``r = 0``.  Higher-``l`` shells pick up an
    extra ``2 l · R'/r`` term in the Laplacian of ``R·Y``;
    the assembly in :func:`cgs_sph_vgl` applies that shift
    per shell.

    ``r`` is accepted purely for API symmetry with the
    cusp-corrected sibling :func:`evaluate_cusp_s_vgl` — it
    does not appear in the expressions below.
    """
    del r  # unused; kept for API symmetry
    e = jnp.exp(-alpha * r2) * norm
    R = jnp.sum(e)
    R_p_over_r = jnp.sum(-2.0 * alpha * e)
    R_lap = jnp.sum(
        (4.0 * alpha**2 * r2 - 6.0 * alpha) * e
    )
    return R, R_p_over_r, R_lap


class _PsiGTO:
    """
    Holds all JAX-compiled wavefunction and energy functions
    derived from a PySCF mean-field object.
    """

    def __init__(self, mf, params_cusp=None, trial=None,
                 jastrow_config=None):
        self.mf = mf
        self.trial = trial
        self._parse_mol(mf, params_cusp, trial)
        self._parse_shells(mf)
        self._parse_elements(mf)
        self._parse_bspline_cfg(jastrow_config)
        self._parse_eeI_cfg(jastrow_config)
        self._build_ao_fns()
        self._build_slater_fns()
        self._build_pade_jastrow_fns()
        self._build_bspline_jastrow_fns()
        self._build_eeI_jastrow_fns()
        self._build_trial_wf_fns()
        self._build_coulomb_fns()
        self._build_ke_fns()

    def _parse_mol(self, mf, params_cusp, trial):
        """Extract MO coefficients, charges, and cusp arrays."""
        mol = mf.mol
        nocc = jnp.count_nonzero(mf.mo_occ > 0)
        if trial is not None:
            import numpy as np
            self.mo_coeff_full = jnp.array(
                np.asarray(trial['mo_coeff'])
            )
            self.ci_coeffs = trial['ci_coeffs']
            self.occ_up = trial['occ_up']
            self.occ_dn = trial['occ_dn']
            self.mo_occ_coeff = self.mo_coeff_full[
                :, self.occ_up[0]
            ]
        else:
            self.mo_occ_coeff = mf.mo_coeff[:, :nocc]

        self.Z_charges = mol.atom_charges()
        self.nelec = mol.tot_electrons()
        self.e_charges = -jnp.ones((self.nelec,))
        self.l_cgto = params_cusp is not None

        if self.l_cgto:
            self.Z_rc = jnp.array(
                [0.1 if Z == 1 else 0.2
                 for Z in self.Z_charges]
            )
            self.Z_cgao_q0 = jnp.array(
                [params_cusp[mol.atom_symbol(i)]['q0']
                 for i in range(mol.natm)]
            )
            self.Z_cgao_coeff = jnp.array(
                [params_cusp[mol.atom_symbol(i)]['coeff']
                 for i in range(mol.natm)]
            )
        else:
            self.Z_rc = jnp.array([])
            self.Z_cgao_q0 = jnp.array([])
            self.Z_cgao_coeff = jnp.array([])

    def _parse_shells(self, mf):
        """Build shell_list from basis, assigning is_cusp flags."""
        mol = mf.mol
        nsgs = 0
        ncgs = 0
        shell_list = []
        for ia, atom in enumerate(mol._atom):
            symb = atom[0]
            basis = mol._basis[symb]
            for ish, ish_basis in enumerate(basis):
                shells = read_shell(ish_basis, ia, nsgs, ncgs)
                for jsh, shell in enumerate(shells):
                    shell.is_cusp = (
                        self.l_cgto and ish == 0 and jsh == 0
                    )
                    nsgs = nsgs + shell.nsgs
                    ncgs = ncgs + shell.ncgs
                    shell_list.append(shell)
        self.shell_list = shell_list

    def _parse_elements(self, mf):
        """Build unique element list and per-atom element index."""
        mol = mf.mol
        unique_elements = []
        atom_to_elem_idx = []
        for ia in range(mol.natm):
            sym = mol.atom_symbol(ia)
            if sym not in unique_elements:
                unique_elements.append(sym)
            atom_to_elem_idx.append(
                unique_elements.index(sym)
            )
        self.unique_elements = unique_elements
        self.atom_to_elem_idx = jnp.array(atom_to_elem_idx)

    def _parse_bspline_cfg(self, jastrow_config):
        """Extract r_cut values from jastrow_config dict."""
        self._bs_j2_cfg = None
        self._bs_j1_cfgs = {}
        if jastrow_config is None:
            for sym in self.unique_elements:
                self._bs_j1_cfgs[sym] = {"r_cut": 10.0}
            self._bs_j2_cfg = {"r_cut": 10.0}
            # default cutoff at 10 bohrs
            return

        if "J2" in jastrow_config:
            r_cut = float(jastrow_config["J2"]["r_cut"])
            self._bs_j2_cfg = {"r_cut": r_cut}
        if "J1" in jastrow_config:
            for sym in self.unique_elements:
                if sym in jastrow_config["J1"]:
                    rc = float(
                        jastrow_config["J1"][sym]["r_cut"]
                    )
                    self._bs_j1_cfgs[sym] = {"r_cut": rc}

    def _parse_eeI_cfg(self, jastrow_config):
        """Extract eeI (three-body) Jastrow config."""
        self._eeI_cfg = {
            "N_eI": 3, "N_ee": 3, "r_cut": 5.0
        }
        if (jastrow_config is not None
                and "J3" in jastrow_config):
            j3 = jastrow_config["J3"]
            for k in ("N_eI", "N_ee"):
                if k in j3:
                    self._eeI_cfg[k] = int(j3[k])
            if "r_cut" in j3:
                self._eeI_cfg["r_cut"] = float(
                    j3["r_cut"]
                )

    def _build_ao_fns(self):
        """Build AO/MO evaluation functions."""
        shell_list = self.shell_list
        # l_cgto = self.l_cgto
        Z_rc = self.Z_rc
        Z_charges = self.Z_charges
        Z_cgao_q0 = self.Z_cgao_q0
        Z_cgao_coeff = self.Z_cgao_coeff
        mo_occ_coeff = self.mo_occ_coeff

        @jax.jit
        def cgs_sph_get(elec_crds, nuc_crds):
            """Spherical GTO evaluation."""
            n_nuc = nuc_crds.shape[0]
            shell_nsgs_total = (
                shell_list[-1].isgs + shell_list[-1].nsgs
            )
            ao_val = jnp.zeros((n_nuc, shell_nsgs_total))
            ao_val_s = jnp.zeros((n_nuc, shell_nsgs_total))
            for i, shell in enumerate(shell_list):
                pos = nuc_crds[shell.iat]
                dr = elec_crds - pos
                r2 = jnp.sum(dr * dr, axis=-1)
                r = jnp.sqrt(r2)
                alpha = shell.alpha
                norm = shell.norm
                rad_s = jnp.sum(
                    jnp.exp(-alpha * r2) * norm
                )
                cgs = _angular_cartesian(shell.am, dr, rad_s)
                if shell.am == 0 and shell.is_cusp:
                    cgs = evaluate_cusp_s(
                        r,
                        Z_rc[shell.iat],
                        Z_charges[shell.iat],
                        rad_s,
                        Z_cgao_q0[shell.iat],
                        Z_cgao_coeff[shell.iat]
                    )
                if shell.am == 0:
                    ao_val_s = ao_val_s.at[
                        shell.iat,
                        shell.isgs:shell.isgs+shell.nsgs
                    ].set(cgs)
                ao_val = ao_val.at[
                    shell.iat,
                    shell.isgs:shell.isgs+shell.nsgs
                ].set(cgs)
            return ao_val, ao_val_s

        @jax.jit
        def get_psi_mo(elec_crds, nuc_crds):
            """Molecular orbital evaluation."""
            ao_val, ao_val_s = jax.vmap(
                cgs_sph_get, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            mo_val = jnp.einsum(
                'ena,am->em', ao_val, mo_occ_coeff
            )
            mo_val_s = jnp.einsum(
                'ena,am->nem', ao_val_s, mo_occ_coeff
            )
            return mo_val, mo_val_s

        @jax.jit
        def cgs_sph_vgl(elec_crds, nuc_crds):
            """Spherical GTO value, gradient, and Laplacian.

            Single-electron evaluator.  Returns the triple

                (ao_val, ao_grad, ao_lap)

            with shapes

                ao_val:  (n_nuc, n_sgs)
                ao_grad: (n_nuc, n_sgs, 3)
                ao_lap:  (n_nuc, n_sgs)

            where ``ao_grad[A, m, :]`` is the spatial gradient
            of AO ``m`` on atom ``A`` with respect to the
            electron coordinate, and ``ao_lap[A, m]`` is its
            Laplacian.  Angular Laplacians vanish
            identically (solid harmonics), so the shell's
            full Laplacian is ``(R'' + 2(l+1) R'/r) · Y``.
            """
            n_nuc = nuc_crds.shape[0]
            nsgs_total = (
                shell_list[-1].isgs + shell_list[-1].nsgs
            )
            ao_val = jnp.zeros((n_nuc, nsgs_total))
            ao_grad = jnp.zeros((n_nuc, nsgs_total, 3))
            ao_lap = jnp.zeros((n_nuc, nsgs_total))
            for shell in shell_list:
                pos = nuc_crds[shell.iat]
                dr = elec_crds - pos
                r2 = jnp.sum(dr * dr, axis=-1)
                r = jnp.sqrt(r2)
                R, R_p_over_r, R_lap = _radial_vgl(
                    shell.alpha, shell.norm, r2, r,
                )
                am = shell.am

                if am == 0 and shell.is_cusp:
                    f_val, f_p_over_r, f_lap = (
                        evaluate_cusp_s_vgl(
                            r,
                            Z_rc[shell.iat],
                            Z_charges[shell.iat],
                            R, R_p_over_r, R_lap,
                            Z_cgao_q0[shell.iat],
                            Z_cgao_coeff[shell.iat],
                        )
                    )
                    val_store = jnp.array([f_val])
                    grad_store = (f_p_over_r * dr)[None, :]
                    lap_store = jnp.array([f_lap])
                elif am == 0:
                    val_store = jnp.array([R])
                    grad_store = (R_p_over_r * dr)[None, :]
                    lap_store = jnp.array([R_lap])
                else:
                    Y, grad_Y = _angular_cartesian_vgl(am, dr)
                    val_store = R * Y
                    grad_store = (
                        R_p_over_r * dr[None, :] * Y[:, None]
                        + R * grad_Y
                    )
                    lap_store = (
                        R_lap + 2.0 * am * R_p_over_r
                    ) * Y

                sl = slice(
                    shell.isgs, shell.isgs + shell.nsgs,
                )
                ao_val = ao_val.at[shell.iat, sl].set(
                    val_store
                )
                ao_grad = ao_grad.at[shell.iat, sl, :].set(
                    grad_store
                )
                ao_lap = ao_lap.at[shell.iat, sl].set(
                    lap_store
                )
            return ao_val, ao_grad, ao_lap

        @jax.jit
        def get_psi_mo_vgl(elec_crds, nuc_crds):
            """MO value, gradient, and Laplacian per electron.

            Returns ``(mo_val, mo_grad, mo_lap)`` with shapes

                mo_val:  (n_e, n_mo)
                mo_grad: (n_e, n_mo, 3)
                mo_lap:  (n_e, n_mo)

            where each electron's MOs are evaluated at its own
            coordinate — ``mo_grad[e, m, :]`` is ∂φ_m/∂r_e
            and ``mo_lap[e, m]`` is ∇²_e φ_m(r_e).
            """
            ao_val, ao_grad, ao_lap = jax.vmap(
                cgs_sph_vgl, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            mo_val = jnp.einsum(
                'ena,am->em', ao_val, mo_occ_coeff
            )
            mo_grad = jnp.einsum(
                'enax,am->emx', ao_grad, mo_occ_coeff
            )
            mo_lap = jnp.einsum(
                'ena,am->em', ao_lap, mo_occ_coeff
            )
            return mo_val, mo_grad, mo_lap

        @jax.jit
        def get_ao_val(elec_crds, nuc_crds):
            """AO evaluation only (no MO contraction)."""
            ao_val, _ = jax.vmap(
                cgs_sph_get, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            return ao_val

        @jax.jit
        def log_slater_det_C(elec_crds, nuc_crds, C):
            """Slater determinant with explicit MO coefficients."""
            ao_val, _ = jax.vmap(
                cgs_sph_get, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            mo_val = jnp.einsum('ena,am->em', ao_val, C)
            alpha_matrix = mo_val[::2, :]
            beta_matrix = mo_val[1::2, :]
            _, log_det_alpha = jnp.linalg.slogdet(
                alpha_matrix
            )
            _, log_det_beta = jnp.linalg.slogdet(
                beta_matrix
            )
            return log_det_alpha + log_det_beta

        @jax.jit
        def log_slater_det_analytic_C(elec_crds, nuc_crds, C):
            """Analytical (log|det S|, ∇_e, ∇²_e) for CPHF path.

            Same as :func:`log_slater_det_analytic` below, but
            contracts AO derivatives with an explicit MO
            coefficient matrix ``C`` instead of the stored
            ``mo_occ_coeff``.  Used by the orbital-response
            (CPHF) branch of :func:`local_energy_ke_C`.
            """
            ao_val, ao_grad, ao_lap = jax.vmap(
                cgs_sph_vgl, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            mo_val = jnp.einsum('ena,am->em', ao_val, C)
            mo_grad = jnp.einsum('enax,am->emx', ao_grad, C)
            mo_lap = jnp.einsum('ena,am->em', ao_lap, C)
            return _slater_det_assemble(
                mo_val, mo_grad, mo_lap,
            )

        self.cgs_sph_get = cgs_sph_get
        self.get_psi_mo = get_psi_mo
        self.cgs_sph_vgl = cgs_sph_vgl
        self.get_psi_mo_vgl = get_psi_mo_vgl
        self.get_ao_val = get_ao_val
        self._log_slater_det_C = log_slater_det_C
        self._log_slater_det_analytic_C = (
            log_slater_det_analytic_C
        )

    def _build_slater_fns(self):
        """Build single- or multi-determinant Slater functions."""
        get_psi_mo = self.get_psi_mo
        get_psi_mo_vgl = self.get_psi_mo_vgl
        cgs_sph_get = self.cgs_sph_get

        @jax.jit
        def log_slater_determinant(elec_crds, nuc_crds):
            """Single-determinant Slater."""
            mo_val, _ = get_psi_mo(elec_crds, nuc_crds)
            alpha_matrix = mo_val[::2, :]
            beta_matrix = mo_val[1::2, :]
            _, log_det_alpha = jnp.linalg.slogdet(
                alpha_matrix
            )
            _, log_det_beta = jnp.linalg.slogdet(
                beta_matrix
            )
            return log_det_alpha + log_det_beta

        @jax.jit
        def log_slater_det_analytic(elec_crds, nuc_crds):
            """Analytical (log|det|, grad_e, lap_e) triple.

            Closed-form ∇_e and ∇²_e of ``log|det S|`` via
            the Slater identity — no autodiff over the
            determinant.  Used by the analytical path of
            :func:`local_energy_ke`.  Returns:

                log_val : scalar
                grad_e  : (n_e, 3)  — ∇_e log|det|
                lap_e   : (n_e,)    — ∇²_e log|det|
            """
            mo_val, mo_grad, mo_lap = get_psi_mo_vgl(
                elec_crds, nuc_crds,
            )
            return _slater_det_assemble(
                mo_val, mo_grad, mo_lap,
            )

        if self.trial is not None:
            mo_coeff_full = self.mo_coeff_full
            ci_coeffs = self.ci_coeffs
            occ_up = self.occ_up
            occ_dn = self.occ_dn

            @jax.jit
            def log_slater_multidet(elec_crds, nuc_crds):
                """Multi-determinant Slater using log-sum-exp."""
                ao_val, _ = jax.vmap(
                    cgs_sph_get, in_axes=(0, None)
                )(elec_crds, nuc_crds)
                mo_val = jnp.einsum(
                    'ena,am->em', ao_val, mo_coeff_full
                )
                mo_alpha = mo_val[::2, :]
                mo_beta = mo_val[1::2, :]

                def eval_one_det(occ_a, occ_b):
                    sign_a, logdet_a = jnp.linalg.slogdet(
                        mo_alpha[:, occ_a]
                    )
                    sign_b, logdet_b = jnp.linalg.slogdet(
                        mo_beta[:, occ_b]
                    )
                    return sign_a, logdet_a, sign_b, logdet_b

                signs_a, logdets_a, signs_b, logdets_b = \
                    jax.vmap(eval_one_det)(occ_up, occ_dn)

                log_abs_dets = logdets_a + logdets_b
                phase_signs = (
                    signs_a * signs_b
                    * jnp.sign(ci_coeffs)
                )
                log_abs_ci = jnp.log(jnp.abs(ci_coeffs))
                log_contributions = log_abs_ci + log_abs_dets
                logmax = jnp.max(log_contributions)
                sum_val = jnp.sum(
                    phase_signs
                    * jnp.exp(log_contributions - logmax)
                )
                return jnp.log(jnp.abs(sum_val)) + logmax

            self._log_slater = log_slater_multidet
            # Multi-det analytical path is a v2 follow-up;
            # the KE driver falls back to linearize for
            # self.trial is not None.
            self._log_slater_analytic = None
        else:
            self._log_slater = log_slater_determinant
            self._log_slater_analytic = log_slater_det_analytic

    def _build_pade_jastrow_fns(self):
        """Build Padé-form Jastrow functions J2_aa, J2_ab, J1."""
        l_cgto = self.l_cgto
        unique_elements = self.unique_elements
        atom_to_elem_idx = self.atom_to_elem_idx
        Z_charges = self.Z_charges

        @jax.jit
        def J2_aa(elec_crds, curr_params):
            """Two-body Jastrow, like-spin pairs (Padé)."""
            i, j = jnp.triu_indices(elec_crds.shape[0], k=1)
            diffs = elec_crds[i] - elec_crds[j]
            r_ij = jnp.sqrt(jnp.sum(diffs*diffs, axis=-1))
            same_spin_mask = (i % 2) == (j % 2)
            same_spin_mask_f = same_spin_mask.astype(
                r_ij.dtype
            )
            u_pairs = curr_params["like"][0] * r_ij \
                / (1. + curr_params["like"][1] * r_ij)
            return jnp.sum(u_pairs * same_spin_mask_f)

        @jax.jit
        def J2_ab(elec_crds, curr_params):
            """Two-body Jastrow, opposite-spin pairs (Padé)."""
            i, j = jnp.triu_indices(elec_crds.shape[0], k=1)
            diffs = elec_crds[i] - elec_crds[j]
            r_ij = jnp.sqrt(jnp.sum(diffs*diffs, axis=-1))
            opp_spin_mask = (i % 2) != (j % 2)
            opp_spin_mask_f = opp_spin_mask.astype(r_ij.dtype)
            u_pairs = curr_params["unlike"][0] * r_ij \
                / (1. + curr_params["unlike"][1] * r_ij)
            return jnp.sum(u_pairs * opp_spin_mask_f)

        @jax.jit
        def J1(elec_crds, nuc_crds, curr_params):
            """One-body Jastrow (Padé)."""
            if l_cgto:
                ab_arr = jnp.stack(
                    [curr_params[sym]
                     for sym in unique_elements]
                )
                a_per_atom = ab_arr[atom_to_elem_idx, 0]
                b_per_atom = ab_arr[atom_to_elem_idx, 1]
            else:
                b_arr = jnp.array(
                    [curr_params[sym]
                     for sym in unique_elements]
                )
                b_per_atom = b_arr[atom_to_elem_idx]
                a_per_atom = -Z_charges
            diffs = (
                elec_crds[None, :, :]
                - nuc_crds[:, None, :]
            )
            r = jnp.linalg.norm(diffs, axis=-1)
            u_vals = a_per_atom[:, None] * r \
                / (1.0 + b_per_atom[:, None] * r)
            return jnp.sum(u_vals)

        @jax.jit
        def J2_pade_vgl(elec_crds, curr_params):
            """Per-electron (grad, lap) of Padé two-body log J₂.

            Returns ``(grad_e, lap_e)`` with shapes
            ``(n_e, 3)`` and ``(n_e,)``, summing the like-
            and unlike-spin contributions.  For each pair
            ``(i, j)`` with ``i < j`` and spin-selected
            ``(a, b)``,

                u(r)  = a r / (1 + b r)
                u'(r) = a / (1 + b r)²
                u''   = -2 a b / (1 + b r)³

            Pair ``(i, j)`` contributes
            ``+u'(r)·diff/r`` to electron ``i`` and the
            opposite to ``j``; the per-electron Laplacian
            contribution ``u'' + 2 u'/r`` lands on both.
            """
            n_e = elec_crds.shape[0]
            i, j = jnp.triu_indices(n_e, k=1)
            diffs = elec_crds[i] - elec_crds[j]
            r_ij = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1))
            same = (i % 2) == (j % 2)
            a_like = curr_params["like"][0]
            b_like = curr_params["like"][1]
            a_unlike = curr_params["unlike"][0]
            b_unlike = curr_params["unlike"][1]
            a = jnp.where(same, a_like, a_unlike)
            b = jnp.where(same, b_like, b_unlike)
            denom = 1.0 + b * r_ij
            up = a / (denom * denom)
            upp = -2.0 * a * b / (denom * denom * denom)
            grad_pair = (up / r_ij)[:, None] * diffs
            lap_pair = upp + 2.0 * up / r_ij

            grad_e = jnp.zeros((n_e, 3), dtype=elec_crds.dtype)
            grad_e = grad_e.at[i].add(grad_pair)
            grad_e = grad_e.at[j].add(-grad_pair)
            lap_e = jnp.zeros(n_e, dtype=elec_crds.dtype)
            lap_e = lap_e.at[i].add(lap_pair)
            lap_e = lap_e.at[j].add(lap_pair)
            return grad_e, lap_e

        @jax.jit
        def J1_pade_vgl(elec_crds, nuc_crds, curr_params):
            """Per-electron (grad, lap) of Padé one-body log J₁.

            Returns ``(grad_e, lap_e)`` with shapes
            ``(n_e, 3)`` and ``(n_e,)``.  Same parameter
            convention as :func:`J1` above:

              * ``l_cgto=True``: ``curr_params[sym]`` is
                ``[a, b]`` per element.
              * ``l_cgto=False``: ``curr_params[sym]`` is
                just ``b``; ``a`` is fixed at ``-Z``.
            """
            if l_cgto:
                ab_arr = jnp.stack([
                    curr_params[sym]
                    for sym in unique_elements
                ])
                a_per_atom = ab_arr[atom_to_elem_idx, 0]
                b_per_atom = ab_arr[atom_to_elem_idx, 1]
            else:
                b_arr = jnp.array([
                    curr_params[sym]
                    for sym in unique_elements
                ])
                b_per_atom = b_arr[atom_to_elem_idx]
                a_per_atom = -Z_charges
            diffs = (
                elec_crds[:, None, :] - nuc_crds[None, :, :]
            )
            r = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1))
            denom = 1.0 + b_per_atom[None, :] * r
            up = a_per_atom[None, :] / (denom * denom)
            upp = (
                -2.0 * a_per_atom[None, :]
                * b_per_atom[None, :]
                / (denom * denom * denom)
            )
            grad_e = jnp.sum(
                (up / r)[..., None] * diffs, axis=1,
            )
            lap_e = jnp.sum(upp + 2.0 * up / r, axis=1)
            return grad_e, lap_e

        self.J2_aa = J2_aa
        self.J2_ab = J2_ab
        self.J1 = J1
        self._J2_pade_vgl = J2_pade_vgl
        self._J1_pade_vgl = J1_pade_vgl

    def _build_bspline_jastrow_fns(self):
        """Build B-spline Jastrow functions."""
        l_cgto = self.l_cgto
        unique_elements = self.unique_elements
        atom_to_elem_idx = self.atom_to_elem_idx
        Z_charges = self.Z_charges
        _bs_j2_cfg = self._bs_j2_cfg
        _bs_j1_cfgs = self._bs_j1_cfgs

        # Precompute per-element J1 cusp values as Python
        # floats so the closure body never has to materialize
        # them under jit.  Without this, the existing
        # ``Z_charges[atom_to_elem_idx == ie][0]`` pattern
        # tracers under jit because ``atom_to_elem_idx`` is a
        # jnp.array.
        _ate_np = np.asarray(atom_to_elem_idx)
        _Z_np = np.asarray(Z_charges)
        _bs_j1_cusp_vals = []
        for _ie, _sym in enumerate(unique_elements):
            if l_cgto:
                _bs_j1_cusp_vals.append(0.0)
            else:
                _bs_j1_cusp_vals.append(
                    -float(_Z_np[_ate_np == _ie][0])
                )

        @jax.jit
        def J2_bspline_aa(elec_crds, curr_params):
            """Two-body B-spline Jastrow, like-spin pairs."""
            n = curr_params["like"].shape[0]
            r_cut = _bs_j2_cfg["r_cut"]
            delta_r = r_cut / (n + 1)
            delta_r_inv = 1.0 / delta_r
            max_index = n
            # Kato cusp: du/dr|_{r->0} = +1/4 (like spin).
            # Note the sign difference from QMCPACK's BsplineFunctor:
            # QMCPACK writes Psi = Det * exp(-J), so its cusp value
            # is -0.25.  Here J is *added* to log|Psi| (exp(+J)
            # convention), so the cusp value must be +0.25.
            coefs = _build_bspline_coefs(
                curr_params["like"], delta_r, 0.25
            )
            i_idx, j_idx = jnp.triu_indices(
                elec_crds.shape[0], k=1
            )
            diffs = elec_crds[i_idx] - elec_crds[j_idx]
            r_ij = jnp.sqrt(
                jnp.sum(diffs * diffs, axis=-1)
            )
            same_spin = (
                (i_idx % 2) == (j_idx % 2)
            ).astype(r_ij.dtype)
            cutoff_mask = (r_ij < r_cut).astype(r_ij.dtype)
            u = _bspline_eval(
                r_ij, coefs, delta_r_inv, max_index
            )
            return jnp.sum(u * same_spin * cutoff_mask)

        @jax.jit
        def J2_bspline_ab(elec_crds, curr_params):
            """Two-body B-spline Jastrow, unlike-spin pairs."""
            n = curr_params["unlike"].shape[0]
            r_cut = _bs_j2_cfg["r_cut"]
            delta_r = r_cut / (n + 1)
            delta_r_inv = 1.0 / delta_r
            max_index = n
            # Kato cusp: du/dr|_{r->0} = +1/2 (unlike spin).
            # Note the sign difference from QMCPACK's BsplineFunctor:
            # QMCPACK writes Psi = Det * exp(-J), so its cusp value
            # is -0.5.  Here J is *added* to log|Psi| (exp(+J)
            # convention), so the cusp value must be +0.5.
            coefs = _build_bspline_coefs(
                curr_params["unlike"], delta_r, 0.5
            )
            i_idx, j_idx = jnp.triu_indices(
                elec_crds.shape[0], k=1
            )
            diffs = elec_crds[i_idx] - elec_crds[j_idx]
            r_ij = jnp.sqrt(
                jnp.sum(diffs * diffs, axis=-1)
            )
            opp_spin = (
                (i_idx % 2) != (j_idx % 2)
            ).astype(r_ij.dtype)
            cutoff_mask = (r_ij < r_cut).astype(r_ij.dtype)
            u = _bspline_eval(
                r_ij, coefs, delta_r_inv, max_index
            )
            return jnp.sum(u * opp_spin * cutoff_mask)

        @jax.jit
        def J1_bspline_fn(elec_crds, nuc_crds, curr_params):
            """One-body B-spline Jastrow."""
            total = 0.0
            for ie, sym in enumerate(unique_elements):
                if sym not in _bs_j1_cfgs:
                    continue
                r_cut = _bs_j1_cfgs[sym]["r_cut"]
                p = curr_params[sym]
                n = p.shape[0]
                delta_r = r_cut / (n + 1)
                delta_r_inv = 1.0 / delta_r
                max_index = n
                cusp_val = _bs_j1_cusp_vals[ie]
                coefs = _build_bspline_coefs(
                    p, delta_r, cusp_val
                )
                elem_mask = (
                    atom_to_elem_idx == ie
                ).astype(elec_crds.dtype)
                diffs = (
                    elec_crds[None, :, :]
                    - nuc_crds[:, None, :]
                )
                r = jnp.linalg.norm(diffs, axis=-1)
                u = _bspline_eval(
                    r.ravel(), coefs, delta_r_inv, max_index
                ).reshape(r.shape)
                cutoff = (r < r_cut).astype(r.dtype)
                total += jnp.sum(
                    u * cutoff * elem_mask[:, None]
                )
            return total

        @jax.jit
        def J2_bspline_aa_vgl(elec_crds, curr_params):
            """Per-electron (grad, lap) of like-spin B-spline J2.

            Returns ``(grad_e, lap_e)`` of shapes ``(n_e, 3)``
            and ``(n_e,)``, mirroring :func:`_J2_pade_vgl`.  The
            cutoff mask is applied to value, gradient and
            Laplacian — by C² continuity at ``r = r_cut`` the
            spline and its first two derivatives vanish there,
            so the masked function remains C².
            """
            n = curr_params["like"].shape[0]
            r_cut = _bs_j2_cfg["r_cut"]
            delta_r = r_cut / (n + 1)
            delta_r_inv = 1.0 / delta_r
            max_index = n
            coefs = _build_bspline_coefs(
                curr_params["like"], delta_r, 0.25
            )
            n_e = elec_crds.shape[0]
            i_idx, j_idx = jnp.triu_indices(n_e, k=1)
            diffs = elec_crds[i_idx] - elec_crds[j_idx]
            r_ij = jnp.sqrt(
                jnp.sum(diffs * diffs, axis=-1)
            )
            same_spin = (
                (i_idx % 2) == (j_idx % 2)
            ).astype(r_ij.dtype)
            cutoff_mask = (r_ij < r_cut).astype(r_ij.dtype)
            mask = same_spin * cutoff_mask
            _, up, upp = _bspline_eval_vgl(
                r_ij, coefs, delta_r_inv, max_index
            )
            grad_pair = (up / r_ij)[:, None] * diffs
            lap_pair = upp + 2.0 * up / r_ij

            grad_e = jnp.zeros(
                (n_e, 3), dtype=elec_crds.dtype
            )
            grad_e = grad_e.at[i_idx].add(
                grad_pair * mask[:, None]
            )
            grad_e = grad_e.at[j_idx].add(
                -grad_pair * mask[:, None]
            )
            lap_e = jnp.zeros(n_e, dtype=elec_crds.dtype)
            lap_e = lap_e.at[i_idx].add(lap_pair * mask)
            lap_e = lap_e.at[j_idx].add(lap_pair * mask)
            return grad_e, lap_e

        @jax.jit
        def J2_bspline_ab_vgl(elec_crds, curr_params):
            """Per-electron (grad, lap) of unlike-spin B-spline J2."""
            n = curr_params["unlike"].shape[0]
            r_cut = _bs_j2_cfg["r_cut"]
            delta_r = r_cut / (n + 1)
            delta_r_inv = 1.0 / delta_r
            max_index = n
            coefs = _build_bspline_coefs(
                curr_params["unlike"], delta_r, 0.5
            )
            n_e = elec_crds.shape[0]
            i_idx, j_idx = jnp.triu_indices(n_e, k=1)
            diffs = elec_crds[i_idx] - elec_crds[j_idx]
            r_ij = jnp.sqrt(
                jnp.sum(diffs * diffs, axis=-1)
            )
            opp_spin = (
                (i_idx % 2) != (j_idx % 2)
            ).astype(r_ij.dtype)
            cutoff_mask = (r_ij < r_cut).astype(r_ij.dtype)
            mask = opp_spin * cutoff_mask
            _, up, upp = _bspline_eval_vgl(
                r_ij, coefs, delta_r_inv, max_index
            )
            grad_pair = (up / r_ij)[:, None] * diffs
            lap_pair = upp + 2.0 * up / r_ij

            grad_e = jnp.zeros(
                (n_e, 3), dtype=elec_crds.dtype
            )
            grad_e = grad_e.at[i_idx].add(
                grad_pair * mask[:, None]
            )
            grad_e = grad_e.at[j_idx].add(
                -grad_pair * mask[:, None]
            )
            lap_e = jnp.zeros(n_e, dtype=elec_crds.dtype)
            lap_e = lap_e.at[i_idx].add(lap_pair * mask)
            lap_e = lap_e.at[j_idx].add(lap_pair * mask)
            return grad_e, lap_e

        @jax.jit
        def J1_bspline_vgl(elec_crds, nuc_crds, curr_params):
            """Per-electron (grad, lap) of one-body B-spline J1.

            Mirrors :func:`J1_bspline_fn` shell loop over unique
            element symbols, accumulating into per-electron
            ``grad_e`` and ``lap_e``.
            """
            n_e = elec_crds.shape[0]
            grad_e = jnp.zeros(
                (n_e, 3), dtype=elec_crds.dtype
            )
            lap_e = jnp.zeros(n_e, dtype=elec_crds.dtype)
            for ie, sym in enumerate(unique_elements):
                if sym not in _bs_j1_cfgs:
                    continue
                r_cut = _bs_j1_cfgs[sym]["r_cut"]
                p = curr_params[sym]
                n = p.shape[0]
                delta_r = r_cut / (n + 1)
                delta_r_inv = 1.0 / delta_r
                max_index = n
                cusp_val = _bs_j1_cusp_vals[ie]
                coefs = _build_bspline_coefs(
                    p, delta_r, cusp_val
                )
                elem_mask = (
                    atom_to_elem_idx == ie
                ).astype(elec_crds.dtype)
                # diffs[a, e, x] = elec[e, x] - nuc[a, x]
                diffs = (
                    elec_crds[None, :, :]
                    - nuc_crds[:, None, :]
                )
                r = jnp.linalg.norm(diffs, axis=-1)
                cutoff = (r < r_cut).astype(r.dtype)
                mask = cutoff * elem_mask[:, None]
                _, up, upp = _bspline_eval_vgl(
                    r.ravel(), coefs, delta_r_inv, max_index
                )
                up = up.reshape(r.shape)
                upp = upp.reshape(r.shape)
                # grad contribution to electron e from atom a
                grad_contrib = (up / r)[..., None] * diffs
                grad_e = grad_e + jnp.sum(
                    grad_contrib * mask[..., None], axis=0,
                )
                lap_contrib = upp + 2.0 * up / r
                lap_e = lap_e + jnp.sum(
                    lap_contrib * mask, axis=0,
                )
            return grad_e, lap_e

        self.J2_bspline_aa = J2_bspline_aa
        self.J2_bspline_ab = J2_bspline_ab
        self.J1_bspline_fn = J1_bspline_fn
        self._J2_bspline_aa_vgl = J2_bspline_aa_vgl
        self._J2_bspline_ab_vgl = J2_bspline_ab_vgl
        self._J1_bspline_vgl = J1_bspline_vgl

    def _build_eeI_jastrow_fns(self):
        """Build the three-body eeI Jastrow closure.

        Ports the ``PolynomialFunctor3D`` from QMCPACK.
        The polynomial order and cutoff are read from
        ``self._eeI_cfg``.
        """
        N_eI = self._eeI_cfg["N_eI"]
        N_ee = self._eeI_cfg["N_ee"]
        r_cut = self._eeI_cfg["r_cut"]
        L = 0.5 * r_cut
        g_shape = (N_eI + 1, N_eI + 1, N_ee + 1)

        A, ls, ms, ns = _build_eeI_constraint_map(
            N_eI, N_ee, r_cut
        )
        unique_elements = self.unique_elements
        atom_to_elem_idx = self.atom_to_elem_idx

        def _eeI_element_spin(
            elec_crds, nuc_crds, gamma_3d,
            ie, same_spin
        ):
            n_elec = elec_crds.shape[0]
            n_atoms = nuc_crds.shape[0]
            i_idx, j_idx = jnp.triu_indices(
                n_elec, k=1
            )
            if same_spin:
                sm = (i_idx % 2) == (j_idx % 2)
            else:
                sm = (i_idx % 2) != (j_idx % 2)
            spin_f = sm.astype(elec_crds.dtype)

            # ee distances (n_pairs,)
            r_12 = jnp.linalg.norm(
                elec_crds[i_idx]
                - elec_crds[j_idx],
                axis=-1,
            )

            # eI distances (n_atoms, n_elec)
            r_eI = jnp.linalg.norm(
                elec_crds[None, :, :]
                - nuc_crds[:, None, :],
                axis=-1,
            )
            r_1I = r_eI[:, i_idx]
            r_2I = r_eI[:, j_idx]

            # masks (n_atoms, n_pairs)
            cutoff = (
                (r_1I < L) & (r_2I < L)
            ).astype(elec_crds.dtype)
            elem = (
                atom_to_elem_idx == ie
            ).astype(elec_crds.dtype)[:, None]
            mask = cutoff * elem * spin_f[None, :]

            # flatten → (n_atoms * n_pairs,)
            n_pairs = i_idx.shape[0]
            r_12_f = jnp.broadcast_to(
                r_12[None, :],
                (n_atoms, n_pairs),
            ).ravel()
            r_1I_f = r_1I.ravel()
            r_2I_f = r_2I.ravel()

            u = _eval_eeI_poly(
                r_12_f, r_1I_f, r_2I_f,
                gamma_3d, L, N_eI, N_ee,
            )
            return jnp.sum(u * mask.ravel())

        def J3_eeI_fn(elec_crds, nuc_crds,
                      curr_params):
            # Sign convention: QMCPACK uses exp(-J),
            # OmegaQMC uses exp(+J).  We negate the
            # polynomial sum so the same parameters
            # carry the same physical meaning.
            total = 0.0
            for ie, sym in enumerate(
                unique_elements
            ):
                for prefix, same in [
                    ("like+", True),
                    ("unlike+", False),
                ]:
                    key = prefix + sym
                    if key not in curr_params:
                        continue
                    gamma_vec = A @ curr_params[key]
                    gamma_3d = _vec_to_gamma_3d(
                        gamma_vec, ls, ms, ns,
                        g_shape,
                    )
                    total += _eeI_element_spin(
                        elec_crds, nuc_crds,
                        gamma_3d, ie, same,
                    )
            return -total

        def _eeI_element_spin_vgl(
            elec_crds, nuc_crds, gamma_3d,
            ie, same_spin,
        ):
            """Per-electron (grad, lap) of one (element,spin)
            block of J3_eeI, *before* the sign flip applied in
            ``J3_eeI_vgl``.
            """
            n_elec = elec_crds.shape[0]
            n_atoms = nuc_crds.shape[0]
            i_idx, j_idx = jnp.triu_indices(n_elec, k=1)
            n_pairs = i_idx.shape[0]

            if same_spin:
                sm = (i_idx % 2) == (j_idx % 2)
            else:
                sm = (i_idx % 2) != (j_idx % 2)
            spin_f = sm.astype(elec_crds.dtype)

            # Difference vectors and distances.
            diff_eI = (
                elec_crds[None, :, :] - nuc_crds[:, None, :]
            )
            diff_1I = diff_eI[:, i_idx, :]
            diff_2I = diff_eI[:, j_idx, :]
            diff_12 = elec_crds[i_idx] - elec_crds[j_idx]
            r_eI = jnp.linalg.norm(diff_eI, axis=-1)
            r_1I = r_eI[:, i_idx]
            r_2I = r_eI[:, j_idx]
            r_12 = jnp.linalg.norm(diff_12, axis=-1)

            # Masks (n_atoms, n_pairs).
            cutoff = (
                (r_1I < L) & (r_2I < L)
            ).astype(elec_crds.dtype)
            elem = (
                atom_to_elem_idx == ie
            ).astype(elec_crds.dtype)[:, None]
            mask = cutoff * elem * spin_f[None, :]

            # Polynomial-× envelope derivatives at every triplet.
            r_12_b = jnp.broadcast_to(
                r_12[None, :], (n_atoms, n_pairs),
            )
            (_u, u_d1I, u_d2I, u_d12,
             u_d1I_1I, u_d2I_2I, u_d12_12,
             u_d1I_12, u_d2I_12) = _eval_eeI_poly_vgl(
                r_12_b.ravel(),
                r_1I.ravel(),
                r_2I.ravel(),
                gamma_3d, L, N_eI, N_ee,
            )
            shp = (n_atoms, n_pairs)
            u_d1I = u_d1I.reshape(shp)
            u_d2I = u_d2I.reshape(shp)
            u_d12 = u_d12.reshape(shp)
            u_d1I_1I = u_d1I_1I.reshape(shp)
            u_d2I_2I = u_d2I_2I.reshape(shp)
            u_d12_12 = u_d12_12.reshape(shp)
            u_d1I_12 = u_d1I_12.reshape(shp)
            u_d2I_12 = u_d2I_12.reshape(shp)

            # Unit vectors.
            ehat_1I = diff_1I / r_1I[..., None]
            ehat_2I = diff_2I / r_2I[..., None]
            ehat_12 = diff_12 / r_12[..., None]

            # Cosines for cross terms.
            cos_1_12 = jnp.einsum(
                'apk,pk->ap', ehat_1I, ehat_12,
            )
            cos_2_12 = jnp.einsum(
                'apk,pk->ap', ehat_2I, ehat_12,
            )

            # Per-triplet grad / lap, masked.
            m_b = mask[..., None]
            ehat_12_b = jnp.broadcast_to(
                ehat_12[None, :, :], (n_atoms, n_pairs, 3),
            )
            grad_i_t = m_b * (
                u_d1I[..., None] * ehat_1I
                + u_d12[..., None] * ehat_12_b
            )
            grad_j_t = m_b * (
                u_d2I[..., None] * ehat_2I
                - u_d12[..., None] * ehat_12_b
            )
            inv_r_12_b = 1.0 / r_12_b
            lap_i_t = mask * (
                u_d1I_1I + 2.0 * u_d1I / r_1I
                + u_d12_12 + 2.0 * u_d12 * inv_r_12_b
                + 2.0 * u_d1I_12 * cos_1_12
            )
            lap_j_t = mask * (
                u_d2I_2I + 2.0 * u_d2I / r_2I
                + u_d12_12 + 2.0 * u_d12 * inv_r_12_b
                - 2.0 * u_d2I_12 * cos_2_12
            )

            # Scatter (n_atoms, n_pairs) triplets onto the
            # n_elec electrons via the pair-electron index map.
            i_flat = jnp.broadcast_to(
                i_idx[None, :], shp,
            ).ravel()
            j_flat = jnp.broadcast_to(
                j_idx[None, :], shp,
            ).ravel()
            grad_e = jnp.zeros(
                (n_elec, 3), dtype=elec_crds.dtype,
            )
            grad_e = grad_e.at[i_flat].add(
                grad_i_t.reshape(-1, 3),
            )
            grad_e = grad_e.at[j_flat].add(
                grad_j_t.reshape(-1, 3),
            )
            lap_e = jnp.zeros(
                (n_elec,), dtype=elec_crds.dtype,
            )
            lap_e = lap_e.at[i_flat].add(lap_i_t.ravel())
            lap_e = lap_e.at[j_flat].add(lap_j_t.ravel())
            return grad_e, lap_e

        @jax.jit
        def J3_eeI_vgl(elec_crds, nuc_crds, curr_params):
            """Per-electron (grad, lap) of log J₃_{eeI}.

            Returns ``(grad_e, lap_e)`` with shapes
            ``(n_e, 3)`` and ``(n_e,)``.  Sign-flipped to
            match the ``-total`` convention of
            :func:`J3_eeI_fn`.
            """
            n_e = elec_crds.shape[0]
            grad_e = jnp.zeros(
                (n_e, 3), dtype=elec_crds.dtype,
            )
            lap_e = jnp.zeros(
                (n_e,), dtype=elec_crds.dtype,
            )
            for ie, sym in enumerate(unique_elements):
                for prefix, same in [
                    ("like+", True),
                    ("unlike+", False),
                ]:
                    key = prefix + sym
                    if key not in curr_params:
                        continue
                    gamma_vec = A @ curr_params[key]
                    gamma_3d = _vec_to_gamma_3d(
                        gamma_vec, ls, ms, ns, g_shape,
                    )
                    g, L_ = _eeI_element_spin_vgl(
                        elec_crds, nuc_crds,
                        gamma_3d, ie, same,
                    )
                    grad_e = grad_e + g
                    lap_e = lap_e + L_
            return -grad_e, -lap_e

        self.J3_eeI_fn = J3_eeI_fn
        self._J3_eeI_vgl = J3_eeI_vgl

    def _build_trial_wf_fns(self):
        """Build log_trial_wavefunction and _C variant."""
        _log_slater = self._log_slater
        _log_slater_det_C = self._log_slater_det_C
        _log_slater_det_analytic_C = (
            self._log_slater_det_analytic_C
        )
        _J1_pade_vgl = self._J1_pade_vgl
        _J2_pade_vgl = self._J2_pade_vgl
        _J1_bspline_vgl = self._J1_bspline_vgl
        _J2_bspline_aa_vgl = self._J2_bspline_aa_vgl
        _J2_bspline_ab_vgl = self._J2_bspline_ab_vgl
        _J3_eeI_vgl = self._J3_eeI_vgl
        trial = self.trial
        J1 = self.J1
        J2_aa = self.J2_aa
        J2_ab = self.J2_ab
        J1_bspline_fn = self.J1_bspline_fn
        J2_bspline_aa = self.J2_bspline_aa
        J2_bspline_ab = self.J2_bspline_ab
        J3_eeI_fn = self.J3_eeI_fn

        @jax.jit
        def log_trial_wavefunction(
            elec_crds, nuc_crds, curr_params
        ):
            """Trial wavefunction."""
            ln_slater = _log_slater(elec_crds, nuc_crds)
            jastrow_term = 0.0
            if "J1_pade" in curr_params:
                jastrow_term += J1(
                    elec_crds, nuc_crds,
                    curr_params["J1_pade"]
                )
            if "J2_pade" in curr_params:
                jastrow_term += J2_aa(
                    elec_crds, curr_params["J2_pade"]
                ) + J2_ab(
                    elec_crds, curr_params["J2_pade"]
                )
            if "J1_bspline" in curr_params:
                jastrow_term += J1_bspline_fn(
                    elec_crds, nuc_crds,
                    curr_params["J1_bspline"]
                )
            if "J2_bspline" in curr_params:
                jastrow_term += J2_bspline_aa(
                    elec_crds, curr_params["J2_bspline"]
                ) + J2_bspline_ab(
                    elec_crds, curr_params["J2_bspline"]
                )
            if "J3_eeI" in curr_params:
                jastrow_term += J3_eeI_fn(
                    elec_crds, nuc_crds,
                    curr_params["J3_eeI"],
                )
            return ln_slater + jastrow_term

        @jax.jit
        def log_trial_wavefunction_C(
            elec_crds, nuc_crds, curr_params, C
        ):
            """Trial wavefunction with explicit MO coefficients."""
            ln_slater = _log_slater_det_C(
                elec_crds, nuc_crds, C
            )
            jastrow_term = 0.0
            if "J1_pade" in curr_params:
                jastrow_term += J1(
                    elec_crds, nuc_crds,
                    curr_params["J1_pade"]
                )
            if "J2_pade" in curr_params:
                jastrow_term += J2_aa(
                    elec_crds, curr_params["J2_pade"]
                ) + J2_ab(
                    elec_crds, curr_params["J2_pade"]
                )
            if "J1_bspline" in curr_params:
                jastrow_term += J1_bspline_fn(
                    elec_crds, nuc_crds,
                    curr_params["J1_bspline"]
                )
            if "J2_bspline" in curr_params:
                jastrow_term += J2_bspline_aa(
                    elec_crds, curr_params["J2_bspline"]
                ) + J2_bspline_ab(
                    elec_crds, curr_params["J2_bspline"]
                )
            if "J3_eeI" in curr_params:
                jastrow_term += J3_eeI_fn(
                    elec_crds, nuc_crds,
                    curr_params["J3_eeI"],
                )
            return ln_slater + jastrow_term

        @jax.jit
        def _local_energy_ke_C_hessian(
            elec_crds, nuc_crds, curr_params, C,
        ):
            """Prior Hessian-based KE(C) — regression only."""
            def _log_psi_flat(p_flat):
                return log_trial_wavefunction_C(
                    p_flat.reshape(-1, 3),
                    nuc_crds, curr_params, C,
                )
            grad_fn = jax.grad(_log_psi_flat)
            hess_fn = jax.hessian(_log_psi_flat)
            p_flat = elec_crds.flatten()
            grad_log_psi = grad_fn(p_flat)
            hess_log_psi = hess_fn(p_flat)
            lap_term = jnp.trace(hess_log_psi)
            grad_term_sq = jnp.sum(grad_log_psi**2)
            return -0.5 * (lap_term + grad_term_sq)

        @jax.jit
        def local_energy_ke_C(
            elec_crds, nuc_crds, curr_params, C,
        ):
            """Analytical forward-Laplacian KE with explicit C.

            The CPHF orbital-response path (`jax.jvp` over
            ``C``) still flows through this kernel, but the
            per-walker working set is now closed-form in
            ``C`` via :func:`_slater_det_assemble`.
            """
            p_flat_shape = elec_crds.shape
            if trial is None:
                _, g_s, L_s = _log_slater_det_analytic_C(
                    elec_crds, nuc_crds, C,
                )
                grad_flat = g_s.reshape(-1)
                lap_total = jnp.sum(L_s)
            else:
                def _slater_flat(p):
                    return _log_slater_det_C(
                        p.reshape(p_flat_shape),
                        nuc_crds, C,
                    )
                lap_s, grad_s = laplacian_linearize(
                    _slater_flat
                )(elec_crds.flatten())
                grad_flat = grad_s
                lap_total = lap_s

            if "J1_pade" in curr_params:
                g, L = _J1_pade_vgl(
                    elec_crds, nuc_crds,
                    curr_params["J1_pade"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)
            if "J2_pade" in curr_params:
                g, L = _J2_pade_vgl(
                    elec_crds, curr_params["J2_pade"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)

            # B-spline J1 / J2 — closed-form VGL.
            if "J1_bspline" in curr_params:
                g, L = _J1_bspline_vgl(
                    elec_crds, nuc_crds,
                    curr_params["J1_bspline"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)
            if "J2_bspline" in curr_params:
                g_aa, L_aa = _J2_bspline_aa_vgl(
                    elec_crds, curr_params["J2_bspline"],
                )
                g_ab, L_ab = _J2_bspline_ab_vgl(
                    elec_crds, curr_params["J2_bspline"],
                )
                grad_flat = (
                    grad_flat
                    + g_aa.reshape(-1)
                    + g_ab.reshape(-1)
                )
                lap_total = (
                    lap_total + jnp.sum(L_aa) + jnp.sum(L_ab)
                )

            # J3_eeI three-body — closed-form VGL.
            if "J3_eeI" in curr_params:
                g, L = _J3_eeI_vgl(
                    elec_crds, nuc_crds,
                    curr_params["J3_eeI"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)

            return -0.5 * (
                lap_total + jnp.sum(grad_flat * grad_flat)
            )

        self.log_trial_wavefunction = log_trial_wavefunction
        self.log_trial_wavefunction_C = log_trial_wavefunction_C
        self.local_energy_ke_C = local_energy_ke_C
        self._local_energy_ke_C_hessian = (
            _local_energy_ke_C_hessian
        )

    def _build_coulomb_fns(self):
        """Build Coulomb interaction energy functions."""
        e_charges = self.e_charges
        Z_charges = self.Z_charges

        @jax.jit
        def classical_coulomb_energy(
            crds1, chgs1, crds2=None, chgs2=None
        ):
            """Coulomb interaction calculation."""
            eps = jnp.finfo(crds1.dtype).eps
            if crds2 is None:
                i, j = jnp.triu_indices(crds1.shape[0], k=1)
                diffs = crds1[i] - crds1[j]
                dists = jnp.sqrt(
                    jnp.sum(diffs * diffs, axis=-1) + eps
                )
                return jnp.sum(chgs1[i] * chgs1[j] / dists)
            elif chgs2 is not None:
                diffs = (
                    crds1[:, None, :] - crds2[None, :, :]
                )
                dists = jnp.sqrt(
                    jnp.sum(diffs * diffs, axis=-1) + eps
                )
                return jnp.sum(
                    chgs1[:, None] * chgs2[None, :] / dists
                )
            else:
                raise ValueError("chgs2 is None")

        @jax.jit
        def local_energy_ee(elec_crds):
            """Electron-electron energy."""
            return classical_coulomb_energy(
                elec_crds, e_charges
            )

        @jax.jit
        def local_energy_nn(nuc_crds):
            """Nuclear-nuclear energy."""
            return classical_coulomb_energy(
                nuc_crds, Z_charges
            )

        @jax.jit
        def local_energy_en(elec_crds, nuc_crds):
            """Electron-nuclear energy."""
            return classical_coulomb_energy(
                elec_crds, e_charges, nuc_crds, Z_charges
            )

        self.classical_coulomb_energy = classical_coulomb_energy
        self.local_energy_ee = local_energy_ee
        self.local_energy_nn = local_energy_nn
        self.local_energy_en = local_energy_en

    def _build_ke_fns(self):
        """Build kinetic energy functions.

        Produces two callables with the same signature:

          * ``local_energy_ke`` — analytical forward-Laplacian
            kernel used in the hot path.  Slater gradient /
            Laplacian via the closed-form
            :func:`_slater_det_assemble`; Padé J1/J2 via the
            ``_J{1,2}_pade_vgl`` helpers; B-spline J1/J2 via
            the matching ``_J{1,2}_bspline*_vgl`` helpers;
            J3_eeI via :func:`J3_eeI_vgl`.  All paths are now
            closed-form; the only remaining
            ``laplacian_linearize`` site is the multi-determinant
            Slater fallback (``trial is not None``).
          * ``_local_energy_ke_hessian`` — the prior
            ``jax.hessian(_log_psi_flat)`` kernel, retained
            verbatim for regression and debug.
        """
        log_trial_wavefunction = self.log_trial_wavefunction
        log_slater_analytic = self._log_slater_analytic
        _log_slater = self._log_slater
        _J1_pade_vgl = self._J1_pade_vgl
        _J2_pade_vgl = self._J2_pade_vgl
        _J1_bspline_vgl = self._J1_bspline_vgl
        _J2_bspline_aa_vgl = self._J2_bspline_aa_vgl
        _J2_bspline_ab_vgl = self._J2_bspline_ab_vgl
        _J3_eeI_vgl = self._J3_eeI_vgl

        @jax.jit
        def _local_energy_ke_hessian(
            elec_crds, nuc_crds, curr_params,
        ):
            """Prior Hessian-based KE kernel (regression only)."""
            def _log_psi_flat(p_flat):
                return log_trial_wavefunction(
                    p_flat.reshape(-1, 3),
                    nuc_crds, curr_params,
                )
            grad_fn = jax.grad(_log_psi_flat)
            hess_fn = jax.hessian(_log_psi_flat)
            p_flat = elec_crds.flatten()
            grad_log_psi = grad_fn(p_flat)
            hess_log_psi = hess_fn(p_flat)
            lap_term = jnp.trace(hess_log_psi)
            grad_term_sq = jnp.sum(grad_log_psi**2)
            return -0.5 * (lap_term + grad_term_sq)

        @jax.jit
        def local_energy_ke(elec_crds, nuc_crds, curr_params):
            """Analytical forward-Laplacian kinetic energy."""
            p_flat_shape = elec_crds.shape
            # Slater: analytical for single-det, linearize
            # fallback for multi-det.
            if log_slater_analytic is not None:
                _, g_s, L_s = log_slater_analytic(
                    elec_crds, nuc_crds,
                )
                grad_flat = g_s.reshape(-1)
                lap_total = jnp.sum(L_s)
            else:
                def _slater_flat(p):
                    return _log_slater(
                        p.reshape(p_flat_shape), nuc_crds,
                    )
                lap_s, grad_s = laplacian_linearize(
                    _slater_flat
                )(elec_crds.flatten())
                grad_flat = grad_s
                lap_total = lap_s

            # Padé J1 / J2 — closed-form VGL.
            if "J1_pade" in curr_params:
                g, L = _J1_pade_vgl(
                    elec_crds, nuc_crds,
                    curr_params["J1_pade"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)
            if "J2_pade" in curr_params:
                g, L = _J2_pade_vgl(
                    elec_crds, curr_params["J2_pade"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)

            # B-spline J1 / J2 — closed-form VGL.
            if "J1_bspline" in curr_params:
                g, L = _J1_bspline_vgl(
                    elec_crds, nuc_crds,
                    curr_params["J1_bspline"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)
            if "J2_bspline" in curr_params:
                g_aa, L_aa = _J2_bspline_aa_vgl(
                    elec_crds, curr_params["J2_bspline"],
                )
                g_ab, L_ab = _J2_bspline_ab_vgl(
                    elec_crds, curr_params["J2_bspline"],
                )
                grad_flat = (
                    grad_flat
                    + g_aa.reshape(-1)
                    + g_ab.reshape(-1)
                )
                lap_total = (
                    lap_total + jnp.sum(L_aa) + jnp.sum(L_ab)
                )

            # J3_eeI three-body — closed-form VGL.
            if "J3_eeI" in curr_params:
                g, L = _J3_eeI_vgl(
                    elec_crds, nuc_crds,
                    curr_params["J3_eeI"],
                )
                grad_flat = grad_flat + g.reshape(-1)
                lap_total = lap_total + jnp.sum(L)

            return -0.5 * (
                lap_total + jnp.sum(grad_flat * grad_flat)
            )

        self._local_energy_ke_hessian = _local_energy_ke_hessian
        self.local_energy_ke = local_energy_ke


def get_psi_fun(mf, params_cusp=None, trial=None,
                jastrow_config=None):
    """
    Creates functions for evaluating the wavefunction
    and local energy components from a PySCF mean-field
    calculation.
    """
    obj = _PsiGTO(
        mf,
        params_cusp=params_cusp,
        trial=trial,
        jastrow_config=jastrow_config,
    )
    return (
        obj.log_trial_wavefunction,
        (obj.local_energy_ee, obj.local_energy_nn,
         obj.local_energy_en, obj.local_energy_ke),
        obj.get_psi_mo,
        (obj.log_trial_wavefunction_C,
         obj.local_energy_ke_C),
    )
