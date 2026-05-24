"""
Neural-network excited-state VMC (NES-VMC) — implementation plan.

This module is a **scaffold** documenting the algorithm we intend to add
to ``OmegaQMC`` for the §VI.B bridge of the methods paper. It is not
yet runnable; building the full SR-with-penalty optimiser inside the
existing ``vmcopt_nn_sr.py`` infrastructure is left to a follow-up work.

Algorithm (Pfau et al. 2024 penalty form)
-----------------------------------------
Train a real-space neural-network wavefunction
:math:`\\Psi_{\\rm NN}^{(k)}` for the :math:`k`-th eigenstate by
optimising

.. math::

    L_k(\\theta_k) \\;=\\;
        E_{\\rm local}\\!\\left[\\Psi_{\\rm NN}^{(k)}\\right]
        \\;+\\; \\lambda \\sum_{j<k}
        \\frac{\\bigl|\\langle\\Psi_{\\rm NN}^{(k)}\\,|\\,\\Psi_{\\rm NN}^{(j)}\\rangle\\bigr|^{2}}
              {\\langle\\Psi_{\\rm NN}^{(k)}|\\Psi_{\\rm NN}^{(k)}\\rangle\\,
               \\langle\\Psi_{\\rm NN}^{(j)}|\\Psi_{\\rm NN}^{(j)}\\rangle}

with :math:`\\Psi_{\\rm NN}^{(j)}` held frozen for :math:`j < k`. The
penalty term forces :math:`\\Psi_{\\rm NN}^{(k)}` orthogonal to all
lower-state trials. The bi-directional Monte Carlo overlap estimator
reads

.. math::

    F_{kj} \\;=\\;
        \\mathbb{E}_{R \\sim |\\Psi_{\\rm NN}^{(k)}|^{2}}\\!
            \\Bigl[\\,\\Psi_{\\rm NN}^{(j)}(R) / \\Psi_{\\rm NN}^{(k)}(R)\\,\\Bigr]
        \\;\\times\\;
        \\mathbb{E}_{R \\sim |\\Psi_{\\rm NN}^{(j)}|^{2}}\\!
            \\Bigl[\\,\\Psi_{\\rm NN}^{(k)}(R) / \\Psi_{\\rm NN}^{(j)}(R)\\,\\Bigr]

The first factor is computed each SR step using the current
:math:`\\Psi_{\\rm NN}^{(k)}` walker bank; the second factor uses the
pre-sampled frozen walker bank from training :math:`\\Psi_{\\rm NN}^{(j)}`.

Pipeline after training
-----------------------
For each state :math:`k`:

1.  Dump a walker bank from :math:`\\Psi_{\\rm NN}^{(k)}` via the
    existing :class:`OmegaQMC.cs.walkers.WalkerDumper` hook.
2.  Apply the standard CS pipeline (no modification needed) to
    recover :math:`\\widehat{c}^{(k)}` in the natural-orbital basis.

Note: the natural orbitals can be those of FCI (as in this paper's
§II.A) or of :math:`\\Psi_{\\rm NN}^{(0)}`; for state-averaged
analysis a common reference frame (e.g. the average NOs of states
0...K) gives the cleanest comparison.

Downstream observables
----------------------

* Transition density matrices
  :math:`\\gamma^{(0k)}_{pq} = \\sum_{IJ} \\widehat{c}_I^{(0)} \\widehat{c}_J^{(k)}
  \\langle D_I | a_p^{\\dagger} a_q | D_J \\rangle`
  follow from standard CI-vector contractions; oscillator strengths
  :math:`f_{0k} = (2/3)\\,\\Delta E_{0k}\\,|\\mu_{0k}|^{2}`
  with :math:`\\mu_{0k} = -\\text{Tr}(\\gamma^{(0k)}\\,r)`.

* State-averaged active-space CASSCF replacement: build the active
  space from union of fractional NOs across states; use the
  state-resolved :math:`\\widehat{c}^{(k)}` as CASCI guesses.

* EOM-CC initialisation: the recovered ground-state
  :math:`\\widehat{c}^{(0)}` already contains the multireference
  character that conventional HF reference EOM-CC struggles with.

Implementation effort
---------------------
Building a working penalty-method NES-VMC in the existing
``vmcopt_nn_sr.py`` framework requires:

* Persisting the trained :math:`\\Psi_{\\rm NN}^{(0)}` parameters and
  a frozen walker bank from training (~50 LOC, mostly checkpoint
  serialisation).
* Evaluating :math:`\\Psi_{\\rm NN}^{(0)} / \\Psi_{\\rm NN}^{(k)}` on
  the live and frozen walker banks each SR step (~100 LOC).
* Modifying the SR loss and gradient to include the penalty term and
  its derivative (~150 LOC, careful with the metric matrix and the
  CG solver).
* Tests, examples, validation (~300 LOC).

Total estimate: ~600 LOC of new code plus several days of training
wall time to demonstrate on \\ce{H2} singlet-singlet vertical
excitation. Deferred to a follow-up paper.
"""

raise NotImplementedError(
    "NES-VMC is documented here but not implemented. See module "
    "docstring for the algorithm and the implementation plan."
)
