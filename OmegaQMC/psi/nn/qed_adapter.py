"""QED adapter bridging NN trial wavefunctions to joint electron-photon evaluation.

Extends :mod:`OmegaQMC.psi.nn.adapter` with a coherent-state-shifted
factorized ansatz for the dipole-gauge Pauli-Fierz Hamiltonian:

.. math::
    \\Psi(\\mathbf r, n) = \\Psi_e(\\mathbf r)\\, \\langle n | \\alpha \\rangle,

equivalently :math:`|\\Psi\\rangle = |\\Psi_e\\rangle\\otimes \\hat D(\\alpha)|0\\rangle`,
where :math:`\\hat D(\\alpha)` is the photon displacement operator and
:math:`\\alpha` is a real learnable scalar absorbing the dominant photon
dressing.

Methodological context
----------------------
Standard QED-CCSD-1 truncates at single photonic excitations from the
vacuum :math:`|0\\rangle`; QED-FCI converges with photon-Fock cutoff
:math:`N_{\\text{ph,max}}`. At ultrastrong coupling
(:math:`\\lambda \\gtrsim 0.5`), both methods need
:math:`N_{\\text{ph,max}} \\gg 10` because the ground state has substantial
photon occupation. The coherent-state shift :math:`\\hat D(\\alpha)` rotates
to a frame where the residual Fock distribution stays narrow around the
displaced vacuum :math:`|\\alpha\\rangle`, dramatically reducing the
effective truncation cost.

This factorized form (electronic NN × photon coherent state) is the
*minimal* coherent-state-shifted ansatz: it lacks explicit
electron-photon entanglement in the NN itself. A follow-on commit will
add :math:`n`-dependence to the NN to capture residual entanglement
beyond the mean coherent-state dressing.

References
----------
- Tang et al. 2025, arXiv:2503.15644 (deep-VMC; non-shifted Fock encoding)
- Haugland et al. 2020, arXiv:2005.04477 (QED-CCSD-1 with coherent-state basis)
- See also ``OmegaQMC.psi.nn.qed_physics`` for the local-energy estimator
  that consumes the ``log_psi`` produced here.
"""
from __future__ import annotations

from typing import Callable, Tuple, Any

import jax
import jax.numpy as jnp
from flax import nnx

from .adapter import make_nn_log_psi
from .qed_physics import coherent_state_log_amplitude


__all__ = [
    "QEDLogPsiParams",
    "make_qed_nn_log_psi",
    "make_qed_nn_log_psi_n_aware",
    "make_qed_nn_log_psi_hybrid",
    "analytical_alpha_perturbative",
    "FockHead",
]


# Type alias for the combined (NN params + alpha) pytree
QEDLogPsiParams = dict[str, Any]   # {'nn': nnx_state, 'alpha': scalar}


def analytical_alpha_perturbative(
    omega: float,
    coupling_vec: jax.Array,
    dipole_expectation: float,
) -> float:
    """First-order analytical estimate of coherent-state shift α.

    Minimizing the dipole-gauge Pauli-Fierz energy at the bilinear-
    coupling level gives the perturbative shift:

    .. math::
        \\alpha = -\\frac{\\lambda\\,\\boldsymbol\\varepsilon\\cdot\\langle\\hat{\\mathbf d}_e\\rangle}{\\sqrt{2\\omega}}

    This cancels the linear forcing term :math:`\\sqrt{\\omega/2}\\,\\lambda\\,
    (\\varepsilon\\cdot\\hat d_e)\\,(b+b^\\dagger)` at the mean dipole.

    Args:
        omega: cavity-mode frequency (Ha).
        coupling_vec: ``(3,)`` light-matter coupling
            (norm = λ, direction = ε).
        dipole_expectation: scalar :math:`\\boldsymbol\\varepsilon\\cdot\\langle\\hat{\\mathbf d}_e\\rangle`
            evaluated at the trial state (e.g., from QED-HF or HF).

    Returns:
        Real scalar α suitable for ``alpha_init``.

    Notes:
        For closed-shell neutral systems with electronic-only dipole
        convention, ``dipole_expectation`` may be zero by symmetry
        (e.g., (H₂)₂ aligned configurations). In that case α=0 is
        already the perturbative optimum and the routine returns 0.
        The learnable α can still adjust at higher orders during VMC
        training.
    """
    cv = jnp.asarray(coupling_vec)
    lam = float(jnp.linalg.norm(cv))
    if lam < 1e-15 or float(omega) < 1e-15:
        return 0.0
    return -lam * float(dipole_expectation) / (2.0 * float(omega)) ** 0.5


def make_qed_nn_log_psi(
    config,
    mol_info,
    rng_key,
    *,
    omega: float,
    coupling_vec: jax.Array,
    alpha_init: float | jax.Array = 0.0,
    alpha_train: bool = True,
) -> Tuple[Callable, QEDLogPsiParams, Any]:
    """Build a coherent-state-shifted joint electron-photon NN ansatz.

    Wraps :func:`make_nn_log_psi` from :mod:`OmegaQMC.psi.nn.adapter`,
    composing the electronic NN log-amplitude with the analytical
    coherent-state log-amplitude :math:`\\log\\langle n | \\alpha \\rangle`.

    The returned ``log_psi`` has signature
    ``(elec_crds, nuc_crds, n, params) -> float``, ready to be consumed
    by :func:`OmegaQMC.psi.nn.qed_physics.pauli_fierz_local_energy`.

    Args:
        config: :class:`NNAnsatzConfig` or string passed to ``make_nn_log_psi``.
        mol_info: :class:`~OmegaQMC.utils.Mole_custom`.
        rng_key: JAX PRNG key for NN parameter init.
        omega: cavity mode frequency (Ha). Stored on params for
            convenience; the local-energy estimator passes it explicitly.
        coupling_vec: ``(3,)`` light-matter coupling vector
            (norm = λ, direction = ε). Same convention as
            :func:`OmegaQMC.qed_fci.qed_fci`.
        alpha_init: initial coherent-state displacement (real scalar).
            Use 0.0 for "start at the photon vacuum". For neutral
            symmetric dimers (e.g., (H₂)₂ along ε_x), the perturbative
            α from :func:`analytical_alpha_perturbative` is zero, so
            the default is fine. For polar systems (water dimer, HF
            dimer) call ``analytical_alpha_perturbative`` first.
        alpha_train: if True (default), α is included in the trainable
            parameter set under the key ``'alpha'`` and updated by the
            optimizer. If False, α is frozen at ``alpha_init`` (test mode).

    Returns:
        Tuple ``(log_psi, init_params, graphdef)``:

        - ``log_psi(elec_crds, nuc_crds, n, params) -> log|Psi(r, n)|``.
          Scalar real output. Differentiable in ``elec_crds`` (kinetic
          energy) and in ``params`` (parameter gradients).
        - ``init_params``: dict ``{'nn': <nnx_state>, 'alpha': <scalar>}``
          or ``{'nn': <nnx_state>}`` if ``alpha_train=False``.
        - ``graphdef``: NNX GraphDef for reconstructing the electronic
          wavefunction model (passed through unchanged from
          ``make_nn_log_psi``).

    Notes:
        * Electronic and photonic factors are independent in this
          ansatz — there is *no* (r, n)-correlation beyond the
          mean-field photon dressing absorbed in α. This is the
          minimal viable coherent-state-shifted ansatz; the planned
          Tang-style n-dependent embedding adds residual correlation.
        * The bilinear-coupling Fock-ladder ratios reduce to closed-form
          coherent-state amplitude ratios:
          :math:`\\Psi(r,n+1)/\\Psi(r,n) = \\alpha/\\sqrt{n+1}`.
          This makes the QED local energy effectively analytical in n
          at leading order — a useful invariant for testing.
    """
    # Build the standard electronic NN log_psi.
    elec_log_psi, elec_init_params, graphdef = make_nn_log_psi(
        config, mol_info, rng_key,
    )

    # Pack initial params. Use plain dict for pytree compatibility.
    alpha0 = jnp.asarray(alpha_init, dtype=jnp.float64)
    if alpha_train:
        init_params: QEDLogPsiParams = {"nn": elec_init_params, "alpha": alpha0}
    else:
        # Frozen α: don't expose to optimizer; close it inside log_psi.
        init_params = {"nn": elec_init_params}

    # Cache the coupling for diagnostic output (not used in evaluation
    # since the local-energy estimator passes omega/coupling_vec explicitly).
    _ = (omega, coupling_vec)  # documented inputs; kept for future autoshift

    if alpha_train:
        def log_psi(elec_crds, nuc_crds, n, params):
            log_elec = elec_log_psi(elec_crds, nuc_crds, params["nn"])
            log_chi = coherent_state_log_amplitude(n, params["alpha"])
            return log_elec + log_chi
    else:
        # α frozen — close over alpha0
        alpha_frozen = alpha0

        def log_psi(elec_crds, nuc_crds, n, params):
            log_elec = elec_log_psi(elec_crds, nuc_crds, params["nn"])
            log_chi = coherent_state_log_amplitude(n, alpha_frozen)
            return log_elec + log_chi

    return log_psi, init_params, graphdef


# ===================================================================
# n-aware ansatz (Phase 2f-1) — Tang-style: NN takes (r, n) jointly.
# ===================================================================

class FockHead(nnx.Module):
    """Small MLP head that produces an n-dependent log correction.

    Input: discrete photon Fock index ``n`` and a vector of global
    r-derived features (e.g. projected dipole ε·d̂_e). Output:
    scalar correction added to the electronic log|Ψ_e(r)|.

    Output layer is zero-initialised, so the head produces a
    correction of exactly 0 at initialisation. This means the joint
    ansatz behaves identically to the standard electronic NN at
    iter 0; the SR optimiser then learns deviations.
    """

    def __init__(
        self,
        nph_max: int,
        n_features: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        self.nph_max = int(nph_max)
        self.embed = nnx.Embed(
            num_embeddings=nph_max + 1,
            features=hidden_dim,
            rngs=rngs,
        )
        self.feature_proj = nnx.Linear(
            in_features=n_features,
            out_features=hidden_dim,
            rngs=rngs,
        )
        # Zero-init output layer: the head adds 0 at init.
        self.out = nnx.Linear(
            in_features=hidden_dim,
            out_features=1,
            use_bias=False,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, n: jax.Array, features: jax.Array) -> jax.Array:
        # n: () int; features: (n_features,) float
        n_emb = self.embed(n)              # (hidden_dim,)
        feat_emb = self.feature_proj(features)  # (hidden_dim,)
        h = jnp.tanh(n_emb + feat_emb)
        return self.out(h)[0]              # scalar


def _projected_dipole(elec_crds: jax.Array, eps_unit: jax.Array) -> jax.Array:
    """ε · d̂_e with electronic dipole d̂_e = -Σ r_i."""
    return -jnp.dot(eps_unit, jnp.sum(elec_crds, axis=0))


def make_qed_nn_log_psi_n_aware(
    config,
    mol_info,
    rng_key,
    *,
    omega: float,
    coupling_vec: jax.Array,
    nph_max: int = 5,
    fock_hidden_dim: int = 64,
) -> Tuple[Callable, QEDLogPsiParams, Any]:
    """Joint (r, n) NN ansatz with output-side n-dependence (Tang-style v1).

    Form:

    .. math::
        \\log \\Psi(r, n) = \\log \\Psi_e(r) + g_\\theta(n;\\,
        \\boldsymbol\\varepsilon \\cdot \\hat{\\mathbf d}_e(r))

    where :math:`g_\\theta` is a small MLP-style head (:class:`FockHead`)
    that takes the discrete photon Fock index ``n`` and a single
    global feature — the projected electronic dipole
    :math:`\\boldsymbol\\varepsilon \\cdot \\hat{\\mathbf d}_e(r)` —
    and returns a scalar correction to the electronic log-amplitude.

    Why this form
    -------------
    The factorized form
    :math:`\\Psi(r,n) = \\Psi_e(r) \\langle n|\\alpha\\rangle`
    has Fock-ladder ratios
    :math:`\\Psi(r,n+1)/\\Psi(r,n) = \\alpha/\\sqrt{n+1}` independent
    of *r*; this zeroes the bilinear-coupling local-energy
    contribution for symmetric systems (Phase 2e finding). With
    :math:`g_\\theta(n;\\,\\varepsilon\\cdot\\hat{\\mathbf d}_e(r))`
    the ratio becomes
    :math:`\\exp[g_\\theta(n+1;f) - g_\\theta(n;f)]` with
    :math:`f = \\varepsilon\\cdot\\hat{\\mathbf d}_e(r)`, i.e. an *r*-dependent
    learnable function. This is sufficient to capture bilinear coupling.

    This is a lightweight approximation to Tang et al. 2025
    (arXiv:2503.15644 §II.C), who append a one-hot encoding of ``n``
    *inside* the per-electron embeddings of the PauliNet2 GNN. Our
    output-side head is less expressive (it sees only the projected
    dipole, not full per-electron features) but does not require
    modifying the existing PsiFormer/PauliNet architecture. A future
    Phase 2f-2 may switch to deeper injection if the head's
    expressivity proves insufficient.

    The head's output layer is zero-initialised: at iter 0 the
    correction is exactly 0 and the joint ansatz reduces to the
    standard electronic NN. SR then learns deviations.

    Args:
        config, mol_info, rng_key: passed through to
            :func:`make_nn_log_psi`.
        omega: cavity-mode frequency (Ha). Stored for diagnostics; the
            head does not directly use ω.
        coupling_vec: ``(3,)`` light-matter coupling
            (norm = λ, direction = ε). The polarisation direction
            defines the projected-dipole feature.
        nph_max: photon Fock cutoff (the head's discrete embedding
            covers ``[0, nph_max]``).
        fock_hidden_dim: hidden width of the Fock head MLP.

    Returns:
        ``(log_psi, init_params, graphdef)``:
          - ``log_psi(elec_crds, nuc_crds, n, params)`` returns scalar
            ``log|Ψ(r, n)|``.
          - ``init_params`` is a dict ``{'nn': <nnx_state>,
            'fock_head': <nnx_state>}``.
          - ``graphdef`` is the electronic-NN graphdef. The Fock
            head's graphdef is captured in the closure (it is small
            enough that exposing it isn't currently needed).
    """
    # 1. Build the standard electronic NN.
    elec_log_psi, elec_init_params, elec_graphdef = make_nn_log_psi(
        config, mol_info, rng_key,
    )

    # 2. Build the Fock head (small MLP).
    rng_key, head_key = jax.random.split(rng_key)
    head_rngs = nnx.Rngs(head_key)
    fock_head = FockHead(
        nph_max=nph_max,
        n_features=1,                  # only projected dipole for now
        hidden_dim=fock_hidden_dim,
        rngs=head_rngs,
    )
    head_graphdef, head_params, head_other = nnx.split(
        fock_head, nnx.Param, ...,
    )

    # 3. Cache the unit polarisation vector.
    cv = jnp.asarray(coupling_vec, dtype=jnp.float64)
    eps_unit = cv / (jnp.linalg.norm(cv) + 1e-30)

    init_params: QEDLogPsiParams = {
        "nn": elec_init_params,
        "fock_head": head_params,
    }

    def log_psi(elec_crds, nuc_crds, n, params):
        log_elec = elec_log_psi(elec_crds, nuc_crds, params["nn"])

        # Global r-feature: projected electronic dipole.
        proj_dip = _projected_dipole(elec_crds, eps_unit)
        feat = jnp.atleast_1d(proj_dip).astype(elec_crds.dtype)

        # Fock-aware correction.
        n_idx = jnp.asarray(n, dtype=jnp.int32)
        head = nnx.merge(head_graphdef, params["fock_head"], head_other)
        delta = head(n_idx, feat).astype(elec_crds.dtype)

        return log_elec + delta

    return log_psi, init_params, elec_graphdef


def make_qed_nn_log_psi_hybrid(
    config,
    mol_info,
    rng_key,
    *,
    omega: float,
    coupling_vec: jax.Array,
    nph_max: int = 5,
    fock_hidden_dim: int = 64,
    alpha_init: float = 0.0,
    alpha_train: bool = False,
) -> Tuple[Callable, QEDLogPsiParams, Any]:
    """Hybrid coherent-state-shifted joint (r, n) NN ansatz (Phase 2f-1, Path B).

    Form:

    .. math::
        \\log \\Psi(r, n) =
            \\log \\Psi_e(r)
            + \\log \\langle n | \\alpha\\rangle
            + g_\\theta(n;\\, \\boldsymbol\\varepsilon \\cdot \\hat{\\mathbf d}_e(r))

    Combines three components:
      1. Standard electronic NN log|Ψ_e(r)|.
      2. **Coherent-state envelope** ⟨n|α⟩ — provides the analytical
         vacuum-centered (or α-centered) Fock distribution as a
         physical prior. At α=0 the envelope is δ_{n,0}; at α≠0 it is
         the Poisson distribution centered at |α|².
      3. **n-aware Fock head** g_θ(n; ε·d_e(r)) — adds learnable
         r-dependent corrections that capture (r, n) entanglement
         beyond the analytical envelope.

    The envelope sets the right *photon* prior so SR doesn't need to
    discover the vacuum from a uniform Fock distribution (which would
    take ~10⁴ iterations as in Tang et al. 2025). The Fock head's
    output layer is zero-initialised, so at iter 0 the joint ansatz
    reduces to ``factorized * envelope`` (same as Phase 2b's form).
    SR then learns the head's deviations.

    This is the architecture our locked-thesis paper targets: the
    envelope gives **efficient Fock representation** at ultrastrong
    coupling, the Fock head gives **expressive (r, n) entanglement**
    at all coupling strengths.

    Args:
        config, mol_info, rng_key: passed to :func:`make_nn_log_psi`.
        omega: cavity-mode frequency (Ha).
        coupling_vec: ``(3,)`` light-matter coupling
            (norm = λ, direction = ε).
        nph_max: photon Fock cutoff. With the coherent-state shift,
            5 is typically sufficient even at λ ~ 1; without shift,
            may need ≥ 30 at large λ.
        fock_hidden_dim: hidden width of the Fock head MLP.
        alpha_init: initial coherent-state displacement. For
            symmetric systems α=0 is the perturbative optimum;
            for polar systems use :func:`analytical_alpha_perturbative`.
        alpha_train: if True, α is included in trainable params.

    Returns:
        ``(log_psi, init_params, graphdef)``:
          - ``log_psi(r, R, n, params) -> log|Ψ(r, n)|`` real scalar.
          - ``init_params`` dict with keys ``'nn'``, ``'fock_head'``,
            and (if ``alpha_train=True``) ``'alpha'``.
          - ``graphdef``: electronic-NN graphdef (head graphdef is
            in the closure).
    """
    # 1. Build the standard electronic NN.
    elec_log_psi, elec_init_params, elec_graphdef = make_nn_log_psi(
        config, mol_info, rng_key,
    )

    # 2. Build the Fock head.
    rng_key, head_key = jax.random.split(rng_key)
    head_rngs = nnx.Rngs(head_key)
    fock_head = FockHead(
        nph_max=nph_max,
        n_features=1,
        hidden_dim=fock_hidden_dim,
        rngs=head_rngs,
    )
    head_graphdef, head_params, head_other = nnx.split(
        fock_head, nnx.Param, ...,
    )

    # 3. Cache unit polarisation and α₀.
    cv = jnp.asarray(coupling_vec, dtype=jnp.float64)
    eps_unit = cv / (jnp.linalg.norm(cv) + 1e-30)
    alpha0 = jnp.asarray(alpha_init, dtype=jnp.float64)

    # 4. Pack params.
    if alpha_train:
        init_params: QEDLogPsiParams = {
            "nn": elec_init_params,
            "fock_head": head_params,
            "alpha": alpha0,
        }
    else:
        init_params = {
            "nn": elec_init_params,
            "fock_head": head_params,
        }

    if alpha_train:
        def log_psi(elec_crds, nuc_crds, n, params):
            log_elec = elec_log_psi(elec_crds, nuc_crds, params["nn"])
            log_chi = coherent_state_log_amplitude(n, params["alpha"])
            proj_dip = _projected_dipole(elec_crds, eps_unit)
            feat = jnp.atleast_1d(proj_dip).astype(elec_crds.dtype)
            n_idx = jnp.asarray(n, dtype=jnp.int32)
            head = nnx.merge(head_graphdef, params["fock_head"], head_other)
            delta = head(n_idx, feat).astype(elec_crds.dtype)
            return log_elec + log_chi + delta
    else:
        alpha_frozen = alpha0

        def log_psi(elec_crds, nuc_crds, n, params):
            log_elec = elec_log_psi(elec_crds, nuc_crds, params["nn"])
            log_chi = coherent_state_log_amplitude(n, alpha_frozen)
            proj_dip = _projected_dipole(elec_crds, eps_unit)
            feat = jnp.atleast_1d(proj_dip).astype(elec_crds.dtype)
            n_idx = jnp.asarray(n, dtype=jnp.int32)
            head = nnx.merge(head_graphdef, params["fock_head"], head_other)
            delta = head(n_idx, feat).astype(elec_crds.dtype)
            return log_elec + log_chi + delta

    return log_psi, init_params, elec_graphdef
