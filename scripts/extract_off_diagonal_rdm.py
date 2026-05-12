"""Off-diagonal 1-RDM estimator: <c_y^dag c_x> for the CH3 e' orbital pair.

The orbital occupation asymmetry of the m_l = +-1 eigenstates is:
   delta_e' = n(e'_+) - n(e'_-) = -2 Im <c_y^dag c_x>
where (c_x, c_y) annihilate electrons in the real e'_x, e'_y MOs.

By the analytical argument, this should also equal <L_z>/hbar in the
dominant-MO approximation. So Im <c_y^dag c_x> is a DIRECT measurement
of the orbital degeneracy lifting.

Algorithm (swap-trick MC estimator):
  For each MC sample R = (r_1, ..., r_N) drawn from |Psi|^2:
    Pick electron e (or sum over all)
    Sample r' from auxiliary distribution Q(r')
    Compute ratio = Psi(R with r_e -> r') / Psi(R)
    Accumulate: phi_y*(r_e) * phi_x(r') * ratio / Q(r')

The estimator converges to <c_y^dag c_x> after averaging.

Practical choices:
  - Q(r') = uniform in a box covering the molecule (simplest)
  - Or Q(r') = |phi_x|^2 (importance sampling — faster convergence)

This script implements the uniform-box version, which is simpler to
code and only ~3-4x slower than importance sampling.

Usage:
    python scripts/extract_off_diagonal_rdm.py <yaml> [--params <pkl>]
        [--n-aux 16] [--box-radius 4]
"""
from __future__ import annotations

import argparse
import math
import os
import os.path as osp
import pickle
import sys
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from pyscf import gto, scf


def _activate_x64():
    jax.config.update("jax_enable_x64", True)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# CH3 reference for orbital coefficients
def build_ch3_mol():
    coords = [["C", (0.0, 0.0, 0.0)]]
    r_ch = 2.039
    for k in range(3):
        theta = 2 * math.pi * k / 3
        coords.append([
            "H", (r_ch * math.cos(theta), r_ch * math.sin(theta), 0.0),
        ])
    return gto.M(
        atom=coords, basis="cc-pVDZ",
        spin=1, unit="Bohr", symmetry=True, verbose=0,
    )


sys.path.insert(0, osp.join(osp.dirname(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to original YAML")
    ap.add_argument("--params", default=None)
    ap.add_argument("--mo-ref",
                    default="scripts/ch3_mo_reference.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-aux", type=int, default=16,
                    help="auxiliary samples per electron-swap")
    ap.add_argument("--box-radius", type=float, default=4.0,
                    help="auxiliary uniform-box half-side in Bohr")
    ap.add_argument("--num-walkers", type=int, default=64,
                    help="MC walkers for the swap-trick eval")
    ap.add_argument("--num-blocks", type=int, default=10,
                    help="MC blocks (each block = full swap-trick pass)")
    args = ap.parse_args()

    _activate_x64()

    cfg = _load_yaml(args.config)
    project = cfg["project"]
    log_dir = osp.join("logs", project)

    params_path = (
        args.params or osp.join(log_dir, f"{project}.params.pkl")
    )
    if not osp.exists(params_path):
        raise FileNotFoundError(f"no params pickle at {params_path}")
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    # Load MO reference
    ref = np.load(args.mo_ref)
    e_x_coef = ref["mo_coeff"][:, ref["e_prime_indices"][0]]
    e_y_coef = ref["mo_coeff"][:, ref["e_prime_indices"][1]]

    # PySCF molecule for AO evaluation at arbitrary positions
    pyscf_mol = build_ch3_mol()

    def eval_phi(positions):
        """Evaluate (phi_x, phi_y) at positions (N_pos, 3) -> (N_pos, 2)."""
        ao = pyscf_mol.eval_gto("GTOval_sph", positions)
        return np.stack([ao @ e_x_coef, ao @ e_y_coef], axis=-1)

    # Set up VMC driver
    from run_qed_vmc import build_molecule
    mol = build_molecule(cfg["system"])
    cav = cfg["cavity"]
    omega = float(cav["omega"])
    lam = float(cav["lambda"])
    eps = jnp.array(cav.get("polarization", [0.0, 0.0, 1.0]),
                    dtype=jnp.float64)
    eps = eps / jnp.linalg.norm(eps)
    coupling_vec = lam * eps
    chiral_eps_y = None
    chiral_handedness = 1
    if "polarization_y" in cav:
        ey = jnp.array(cav["polarization_y"], dtype=jnp.float64)
        chiral_eps_y = ey / jnp.linalg.norm(ey)
        chiral_handedness = int(cav.get("chiral_handedness", 1))

    opt_cfg = cfg["optimizer"]
    seed = int(cfg.get("seed", 0))
    init_key = jax.random.key(seed)
    nph_max = int(opt_cfg.get("nph_max", 8))
    complex_psi_flag = bool(opt_cfg.get("complex_psi", False))
    assert complex_psi_flag, "Off-diagonal 1-RDM only meaningful for complex_psi"

    from OmegaQMC.qed_vmcopt_nn_sr import get_qed_vmcopt_nn_sr_func
    opt = get_qed_vmcopt_nn_sr_func(
        mol, cfg["ansatz"]["type"], init_key,
        omega=omega, coupling_vec=coupling_vec,
        alpha_init=float(opt_cfg.get("alpha_init", 0.0)),
        alpha_train=bool(opt_cfg.get("alpha_train", True)),
        nph_max=nph_max,
        n_aware=bool(opt_cfg.get("n_aware", False)),
        fock_hidden_dim=int(opt_cfg.get("fock_hidden_dim", 64)),
        arch=opt_cfg.get("arch"),
        complex_psi=complex_psi_flag,
        chiral_eps_y=chiral_eps_y,
        chiral_handedness=chiral_handedness,
    )
    opt.driver.params = params

    # The complex-log-psi callable (returns log|Psi| + i*arg(Psi))
    log_psi_signed = opt.driver._log_psi_signed
    n_elec = mol.n_up + mol.n_down
    nuc_crds = opt.driver.nuc_crds

    # not used; see log_psi_batch below

    print(f"running off-diagonal 1-RDM estimator")
    print(f"  config: {args.config}")
    print(f"  lambda={lam}, handedness={chiral_handedness}")
    print(f"  walkers={args.num_walkers}, blocks={args.num_blocks}, "
          f"aux/electron={args.n_aux}")

    # Equilibrate walkers
    key_init = jax.random.key(seed + 5)
    elec, n_ph = opt.driver.initialize_walkers(
        key_init, args.num_walkers,
    )
    mh_step = opt.driver._mh_step_batch
    sigma_r = 0.1
    print(f"  equilibrating walkers...")
    for _ in range(50):
        key_init, key_step = jax.random.split(key_init)
        step_keys = jax.random.split(key_step, args.num_walkers)
        elec, n_ph, _, _, _ = mh_step(
            step_keys, elec, n_ph, params, sigma_r, 0.5,
        )

    # The actual estimator: <phi_y^*(r_e) phi_x(r') (Psi(R')/Psi(R)) V_box>
    # Sum over electrons e and average over walkers + auxiliary draws.
    rng = np.random.default_rng(seed + 11)
    V_box = (2 * args.box_radius) ** 3

    @jax.jit
    def log_psi_batch(elec_batch, n_ph_batch):
        """Returns log|Psi| + i*phase for a batch of configurations.

        log_psi_signed(elec, nuc, n, params) -> (log_mag, sign).
        For complex_psi, sign = psi/|psi| is complex.
        """
        def one(elec_one, n_one):
            log_mag, sign = log_psi_signed(
                elec_one, nuc_crds, n_one, params,
            )
            return log_mag, sign
        log_mags, signs = jax.vmap(one)(elec_batch, n_ph_batch)
        if jnp.iscomplexobj(signs):
            return log_mags.astype(jnp.complex128) + jnp.log(signs)
        else:
            log_signs = jnp.where(
                signs > 0,
                jnp.zeros_like(log_mags, dtype=jnp.complex128),
                jnp.full_like(log_mags, 1j * jnp.pi,
                              dtype=jnp.complex128),
            )
            return log_mags.astype(jnp.complex128) + log_signs

    # Estimator: γ_kl ≈ <Σ_e φ_k*(r_e) · V_box · <φ_l(r') · Ψ(R')/Ψ(R)>_r'>
    # We accumulate the SUM over electrons inside each walker
    # (not the average) — averaging is over walkers only.
    walker_sums_re = []
    walker_sums_im = []

    t0 = datetime.now()
    for blk in range(args.num_blocks):
        # MC equilibrate a few steps between blocks
        for _ in range(10):
            key_init, key_step = jax.random.split(key_init)
            step_keys = jax.random.split(key_step, args.num_walkers)
            elec, n_ph, _, _, _ = mh_step(
                step_keys, elec, n_ph, params, sigma_r, 0.5,
            )

        elec_np = np.asarray(elec)
        n_ph_np = np.asarray(n_ph)
        log_psi_R = np.asarray(log_psi_batch(elec, n_ph))

        # For each walker, accumulate the SUM over electrons.
        for w in range(args.num_walkers):
            walker_sum = 0j
            for e in range(n_elec):
                r_e = elec_np[w, e]
                phi_y_at_re = eval_phi(r_e[None, :])[0, 1]
                r_aux = rng.uniform(
                    -args.box_radius, args.box_radius,
                    size=(args.n_aux, 3),
                )
                phi_x_at_aux = eval_phi(r_aux)[:, 0]
                elec_pert = np.tile(elec_np[w:w+1], (args.n_aux, 1, 1))
                elec_pert[:, e, :] = r_aux
                n_ph_pert = np.tile(n_ph_np[w:w+1], (args.n_aux,))
                log_psi_pert = np.asarray(
                    log_psi_batch(
                        jnp.asarray(elec_pert),
                        jnp.asarray(n_ph_pert),
                    )
                )
                ratio = np.exp(log_psi_pert - log_psi_R[w])
                contrib_e = (
                    phi_y_at_re * phi_x_at_aux * ratio * V_box
                ).mean()
                walker_sum += contrib_e
            walker_sums_re.append(float(np.real(walker_sum)))
            walker_sums_im.append(float(np.imag(walker_sum)))

        elapsed = (datetime.now() - t0).total_seconds()
        running_re = float(np.mean(walker_sums_re))
        running_im = float(np.mean(walker_sums_im))
        running_im_serr = float(
            np.std(walker_sums_im) / np.sqrt(len(walker_sums_im))
        )
        running_delta = -2 * running_im
        print(f"  blk {blk+1}/{args.num_blocks}: "
              f"<c_y* c_x>_re={running_re:+.5f} "
              f"<c_y* c_x>_im={running_im:+.5f} "
              f"(±{running_im_serr:.5f}) "
              f"δ_e'={running_delta:+.5f} "
              f"t={elapsed:.0f}s")

    avg_real = float(np.mean(walker_sums_re))
    avg_imag = float(np.mean(walker_sums_im))
    serr_real = float(np.std(walker_sums_re) / np.sqrt(len(walker_sums_re)))
    serr_imag = float(np.std(walker_sums_im) / np.sqrt(len(walker_sums_im)))
    # Sign convention derivation:
    #   c_+ = (c_x + i c_y)/√2, c_- = (c_x - i c_y)/√2
    #   n_+ - n_- = i (<c_x^† c_y> - <c_y^† c_x>) = +2 Im<c_y^† c_x>
    # So δ_e' = +2 Im(c_y_dag c_x), NOT -2.
    delta_e = +2 * avg_imag
    delta_e_serr = 2 * serr_imag

    out = args.out or osp.join(log_dir, f"{project}.rdm_off_diagonal.npz")
    np.savez(
        out,
        c_yx_real=avg_real, c_yx_imag=avg_imag,
        c_yx_real_serr=serr_real, c_yx_imag_serr=serr_imag,
        delta_e_prime=delta_e, delta_e_prime_serr=delta_e_serr,
        cavity_lambda=lam, chiral_handedness=chiral_handedness,
        n_walker_samples=len(walker_sums_re),
    )
    print(f"\n=== FINAL ===")
    print(f"  <c_y^dag c_x>_real = {avg_real:+.5f} ± {serr_real:.5f}")
    print(f"  <c_y^dag c_x>_imag = {avg_imag:+.5f} ± {serr_imag:.5f}")
    print(f"  delta_e' = n(e'_+) - n(e'_-) = +2 Im<c_y^dag c_x> = "
          f"{delta_e:+.5f} ± {delta_e_serr:.5f}")
    # Predicted from analytical L_z argument:
    # <L_z>_e' = 0.78 (n_+ - n_-) = 0.78 × δ_e' = +1.56 Im<c_y^dag c_x>
    # So predicted Im<c_y^dag c_x> = <L_z> / 1.56.
    print(f"\n  Predicted Im<c_y^dag c_x> = <L_z>/1.56 if all <L_z> from e':")
    print(f"    For <L_z>=+0.053 (L050 sigma+): predict {+0.053/1.56:+.5f}")
    print(f"    For <L_z>=-0.037 (L050 sigma-): predict {-0.037/1.56:+.5f}")
    print(f"  saved to {out}")


if __name__ == "__main__":
    main()
