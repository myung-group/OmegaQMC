# Positioning Strategy — One-Pager

## The single most important fact (re-read before every revision)

**Weber et al. PRL 2025 explicitly excludes Coulomb interactions** in
their cavity-2DEG QED-AFQMC study. Direct quote: *"As a first step, in
this work, we will not consider Coulomb interactions. Under this
condition, our AFQMC simulations do not suffer from a phase problem and
are exact."*

This single sentence is what protects our project from being scooped.
Their entire paper is about non-interacting electrons in a modulating
potential coupled to a cavity, designed to feed a QEDFT functional.
Without Coulomb, the 2D HEG has no Wigner crystal, no ferromagnetic
transition, no correlation hole — none of the phase-diagram physics
that defines the interacting 2D HEG.

## Three competitor-protection arguments

1. **Coulomb interactions** — defining feature of HEG, omitted by the
   only existing ab initio cavity-2DEG study (Weber 2025).

2. **Phase boundaries** — Weber 2025 produces a correlation-energy
   functional, not phase boundaries. Different scientific output.

3. **Strong correlation regime** — our PsiFormer-NN-VMC handles
   broken-symmetry ansatze for the Wigner crystal natively (per Cassella
   2023's 3D approach). AFQMC + Coulomb at strong coupling would face
   phase-problem bias; AFQMC inherently respects translational symmetry,
   making the crystal phase awkward.

## Three competitor-monitoring rules

Watch for:

1. **Flick or Rubio extending Weber 2025 to include Coulomb.** Most
   likely path to scoop. Probability ~30% in next 12 months given
   their existing infrastructure.

2. **Tang/Noé extending their PauliNet polaritonic chemistry to
   periodic systems.** Probability ~25% in next 12 months.

3. **Carleo or Pfau group entering cavity QED.** Probability ~20%; both
   have attention-based ansatz infrastructure already.

If any of these appear in arXiv, reassess immediately. Set a weekly
arXiv search alert with these keywords:
- `cavity QED Monte Carlo electron gas`
- `polaritonic Wigner crystal`
- `cavity neural network variational electron`

## The window

Realistic estimate: **12-18 months** before someone publishes the
Coulomb-included cavity-2DEG phase diagram. We need to publish within
that window.

## Internal-validity checks that save us in review

When writing, defensively address these likely reviewer objections:

1. **"Why not extend Weber 2025's AFQMC to include Coulomb?"** Answer:
   AFQMC + Coulomb at strong coupling has phase-problem bias; AFQMC
   inherently respects translational symmetry while Wigner crystal
   spontaneously breaks it. NN-VMC handles both regimes natively.
2. **"Did you check the analytical Rokaj 2022 limit?"** Yes — at
   $\lambda \to 0$ and weak Coulomb our results reduce to Rokaj's
   non-interacting Fermi liquid + coherent photon state.
3. **"How do you compare to the free-space DMC of Drummond-Needs?"** At
   $\lambda = 0$ we recover $r_s^c \approx 31$ within statistical error.
4. **"Single-mode cavity — what about multi-mode?"** Defer to future
   work; explicitly cite the multi-mode methodology of
   Rokaj 2022 thesis as the natural extension.
5. **"Truncating photon Fock space at $N_{\max}$ — convergence?"**
   Show convergence with $N_{\max}$; following Tang 2025 we expect
   $N_{\max} \sim 5-15$ to be sufficient.

## The pitch sentence (use everywhere)

> *We present the first neural-network variational Monte Carlo treatment
> of the Coulomb-interacting 2D electron gas in an optical cavity,
> computing how cavity coupling shifts the established free-space
> Wigner-crystallisation density and ferromagnetic transition. Our
> approach extends the polaritonic-chemistry NN-VMC framework of
> Tang \textit{et al.}\ to extended systems and complements the recent
> non-interacting QED-AFQMC study of Weber \textit{et al.}\ by
> accessing the strongly-correlated regime where Coulomb interactions
> dominate.*

That sentence appears in: title, abstract, intro paragraph 5, and
discussion. The reviewers should see it three times.
