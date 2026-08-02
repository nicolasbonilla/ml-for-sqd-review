# -*- coding: utf-8 -*-
"""Hard multireference evidence + order-parameter validation for N2 CAS(10e,12o) cc-pVDZ.
For R in {1.10 (near-eq), 2.00 (paper), 2.50 (stretched)}:
  (1) FCI natural-orbital occupation numbers (NOONs) of the active space -> multireference metric.
  (2) Spearman rank correlation between the cheap Epstein-Nesbet prior and exact |c|^2
      -> the paper's proposed order parameter, measured vs geometry.
"""
import json, numpy as np
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1
from scipy.stats import spearmanr

NCAS, NELECAS = 12, (5, 5)
HA2EV = 27.211386
out = {"NCAS": NCAS, "NELECAS": list(NELECAS), "geoms": {}}

for R in (1.10, 1.40, 1.70, 2.00, 2.30, 2.50):
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", verbose=0)
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
    rho = float(spearmanr(w_cheap, w_true).correlation)

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
with open("/w/orderparam.dat", "w") as fh:
    fh.write("R spearman sumfrac mrweight homo_noon lumo_noon\n")
    for k in sorted(out["geoms"], key=float):
        g = out["geoms"][k]
        fh.write(f"{g['R_ang']:.2f} {g['spearman_cheap_vs_true']:.4f} {g['sum_fractional_occ']:.4f} "
                 f"{1-g['weight_HF_determinant']:.4f} {g['frontier_HOMO_NOON']:.4f} {g['frontier_LUMO_NOON']:.4f}\n")
print("WROTE physics_results.json + orderparam.dat")
