# -*- coding: utf-8 -*-
"""Hard multireference evidence + order-parameter validation for N2 CAS(10e,12o) cc-pVDZ.
For R in {1.10 (near-eq), 2.00 (paper), 2.50 (stretched)}:
  (1) FCI natural-orbital occupation numbers (NOONs) of the active space -> multireference metric.
  (2) Spearman rank correlation between the cheap Epstein-Nesbet prior and exact |c|^2
      -> the paper's proposed order parameter, measured vs geometry.

GAUGE (see gauge_study/). mol.symmetry=True fixes the orbital gauge. Without it the
canonical RHF orbitals are free to rotate inside the degenerate pi shells: identical code
returns different orientations at the same energy to 1e-13 Ha, and rho -- which ranks
per-string weights -- moves with the orientation, so the order parameter would not be a
property of the molecule but of the run. Run with OMP_NUM_THREADS=1.

REWARD THRESHOLD. rho is measured with cheap-reward entries below RHO_FLOOR*max set to
zero. Those entries are algebraically zero by the Slater-Condon rules (the triple-and-higher
alpha-strings). Brillouin's theorem does NOT dispose of them at the alpha-string level,
because the reward marginalises over beta: a singly excited alpha-string still carries the
(alpha-single x beta-single) doubles. What survives in the sub-threshold entries is
cancellation residue down to ~1e-42, and Spearman ranks that residue above the exact zeros,
which reads rho high by 0.085 on average across gauges and gives it a spurious gauge spread
of 0.063 (with the cut: 0.034). Measured in gauge_study/rho_sensibilidad.py.

The cut is not a tunable knob because the spectrum is bimodal: the genuine entries begin at
1.2e-12 of the maximum and the residue tops out around 1e-34. It is not literally an empty
band either -- in the worst of 41 gauges two of the 792 entries land between 1e-16 and
1e-12 -- and we say so rather than claiming an emptiness the measurement does not show.
"""
import json, numpy as np
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1
from scipy.stats import spearmanr

NCAS, NELECAS = 12, (5, 5)
HA2EV = 27.211386
RHO_FLOOR = 1e-12          # relative cut on the cheap reward before ranking; see module docstring
out = {"NCAS": NCAS, "NELECAS": list(NELECAS), "geoms": {}}

for R in (1.10, 1.40, 1.70, 2.00, 2.30, 2.50):
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", symmetry=True, verbose=0)
    mf = scf.RHF(mol).run()
    cas = mcscf.CASCI(mf, NCAS, NELECAS)
    h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
    na, nb = NELECAS
    strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
    hf_idx = int(np.where(strs_a == (1 << na) - 1)[0][0])

    # ---- cheap Epstein-Nesbet prior (single H|D_HF> matvec + diagonal) ----
    civ_hf = np.zeros((dim_a, dim_a)); civ_hf[hf_idx, hf_idx] = 1.0
    h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
    Hc = direct_spin1.contract_2e(h2e, civ_hf, NCAS, NELECAS).reshape(dim_a, dim_a)
    hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)
    E_hf = Hc[hf_idx, hf_idx]; denom = E_hf - hdiag; denom[hf_idx, hf_idx] = 1.0
    c1 = Hc / denom; c1[hf_idx, hf_idx] = 1.0
    w_cheap = (c1**2).sum(axis=1)                       # alpha-string marginal (cheap)

    # ---- exact FCI ----
    e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
    civec = civec.reshape(dim_a, dim_a)
    w_true = (civec**2).sum(1)                            # alpha-string marginal (exact)

    # ---- natural-orbital occupations from the FCI 1-RDM ----
    dm1 = pyscf.fci.direct_spin1.make_rdm1(civec, NCAS, NELECAS)  # spin-summed 1-RDM
    noons = np.sort(np.linalg.eigvalsh(dm1))[::-1]        # descending occupations
    # multireference diagnostics
    n_frac = float(np.sum([min(x, 2-x) for x in noons]))  # sum of fractional occupancy
    homo_noon = float(noons[na-1]); lumo_noon = float(noons[na])  # frontier
    # weight of HF determinant in exact wavefunction (single-reference-ness)
    c0_hf = float(civec[hf_idx, hf_idx]); w_hf = c0_hf**2

    # ---- order parameter: Spearman(cheap prior, exact |c|^2) over the alpha strings ----
    # rank the thresholded prior: below RHO_FLOOR*max the entries are algebraic zeros
    # holding cancellation residue only, and ranking residue reads rho high by ~0.085
    w_rank = np.where(w_cheap < RHO_FLOOR * w_cheap.max(), 0.0, w_cheap)
    rho = float(spearmanr(w_rank, w_true).correlation)
    n_zeroed = int((w_rank == 0).sum())        # reported in the .dat header, not hardcoded:
                                               # the count depends on the gauge (it drops as
                                               # symmetry sparsifies the coupling)

    out["geoms"][f"{R:.2f}"] = {
        "R_ang": R, "E_FCI_Ha": float(e_fci),
        "NOONs_top8": [round(float(x), 4) for x in noons[:8]],
        "frontier_HOMO_NOON": round(homo_noon, 4), "frontier_LUMO_NOON": round(lumo_noon, 4),
        "sum_fractional_occ": round(n_frac, 4),
        "weight_HF_determinant": round(w_hf, 4),
        "spearman_cheap_vs_true": round(rho, 4),
    }
    print(f"R={R:.2f}  E_FCI={e_fci:.6f}  HOMO/LUMO NOON={homo_noon:.3f}/{lumo_noon:.3f}  "
          f"sum_frac={n_frac:.3f}  w_HF={w_hf:.3f}  Spearman={rho:.3f}")

json.dump(out, open("/w/physics_results.json", "w"), indent=1)
# emit .dat for the native order-parameter figure: R, spearman, sum_frac, 1-w_HF, homo_noon
# Header commented with % and not #: pgfplots does not honour #, it reads the header as
# data and wrecks the plot without raising anything (from numpy: comments='%').
with open("/w/orderparam.dat", "w") as fh:
    fh.write(
        "% parametro de orden de N2: rho y marcadores de caracter multirreferencial\n"
        f"% N2 CAS({sum(NELECAS)}e,{NCAS}o) cc-pVDZ, CASCI sobre orbitales canonicos RHF\n"
        f"% GAUGE: adaptado por simetria ({mol.groupname}, mol.symmetry=True), OMP_NUM_THREADS=1\n"
        "% rho = Spearman(recompensa Epstein-Nesbet marginalizada, |c|^2 exacto) sobre las\n"
        f"%   {dim_a} cadenas alpha, CON las recompensas por debajo de {RHO_FLOOR:g}*max puestas a cero.\n"
        "%   Sin ese umbral el estadistico rankea residuo de cancelacion (hasta 1e-42) por\n"
        "%   encima de los ceros algebraicos, lo que sube rho en 0.085 de media y le da una\n"
        "%   dispersion de gauge de 0.063 (con umbral: 0.034). Ceros algebraicos: 701 de 792\n"
        "%   -- 546 triples-y-superiores por Slater-Condon mas 155 dobles anuladas por\n"
        "%   simetria espacial; las 91 restantes son las unicas rankeables. No confundir con\n"
        f"%   las {n_zeroed} entradas que quedan por debajo de {RHO_FLOOR:g}*max, que es un conteo\n"
        "%   posterior al umbral (y aqui lleva el valor de la ULTIMA geometria del barrido).\n"
        "%   El corte no es ajustable porque el espectro es bimodal: la senal empieza en\n"
        "%   1.2e-12*max y el residuo no pasa de ~1e-34. Medido en\n"
        "%   gauge_study/rho_sensibilidad.py y figures/cuenta_rankeables.py.\n"
        "% generado por calculations/compute_physics.py\n")
    fh.write("R spearman sumfrac mrweight homo_noon lumo_noon\n")
    for k in sorted(out["geoms"], key=float):
        g = out["geoms"][k]
        fh.write(f"{g['R_ang']:.2f} {g['spearman_cheap_vs_true']:.4f} {g['sum_fractional_occ']:.4f} "
                 f"{1-g['weight_HF_determinant']:.4f} {g['frontier_HOMO_NOON']:.4f} {g['frontier_LUMO_NOON']:.4f}\n")
print("WROTE physics_results.json + orderparam.dat")
