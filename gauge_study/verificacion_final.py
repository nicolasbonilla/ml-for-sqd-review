# -*- coding: utf-8 -*-
"""VERIFICACION FINAL — lo que el paper debe imprimir.

Comprueba por mi cuenta las cuatro afirmaciones que deciden la redaccion:
  A. ¿La casi-degeneracion w(11)=w(12) es EXACTA al apretar la convergencia?
     (si lo es, mi argumento original valia y lo retracte por numeros mal convergidos)
  B. ¿El gauge adaptado por simetria es DETERMINISTA, y que numeros da?
  C. ¿El espectro de Schmidt es invariante de gauge, y cuales son sus conteos?
  D. ¿Que cum(n) son invariantes? (deben serlo los que cierran multipletes completos)

Corrige ademas el bug B1 de mi script anterior: la tolerancia era ABSOLUTA de 1e-12
frente a pesos ~1e-2, asi que el completado de multipletes nunca se disparaba.
"""
import numpy as np
from pyscf import ao2mo, fci, gto, scf, symm

NCAS, NELECAS = 12, (5, 5)
SCF_TOL, FCI_TOL = 1e-12, 1e-13


def solve(symmetry):
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz",
                symmetry=symmetry, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_TOL
    mf.run()
    from pyscf import mcscf
    cas = mcscf.CASCI(mf, NCAS, NELECAS)
    h1, ecore = cas.get_h1cas()
    h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
    s = fci.direct_spin1.FCI()
    s.conv_tol = FCI_TOL
    e, civec = s.kernel(h1, h2, NCAS, NELECAS)
    return mol, mf, float(e + ecore), civec


def counts(w, tol_rel=1e-6):
    """Cruce simple y multiplete COMPLETO, con tolerancia RELATIVA (bug B1 corregido)."""
    cum = np.cumsum(w)
    out = {}
    for thr, lab in ((0.90, "90"), (0.99, "99"), (0.999, "99.9")):
        i = int(np.searchsorted(cum, thr) + 1)
        k = i
        while k < len(w) and abs(w[k] - w[k - 1]) <= tol_rel * abs(w[k - 1]):
            k += 1
        out[lab] = (i, k)
    return out, cum


print("=" * 74)
print("A + B.  GAUGE ADAPTADO POR SIMETRIA (mol.symmetry=True)")
print("=" * 74)
mol, mf, efci, civec = solve(True)
lab = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
w = np.sort((civec ** 2).sum(axis=1))[::-1]
c, cum = counts(w)
print(f"grupo = {mol.groupname}   E_RHF = {mf.e_tot:.12f}   E_FCI = {efci:.12f}")
print(f"irreps activos: {' '.join(str(lab[i]) for i in range(2, 14))}")
print(f"MO5 coef. de referencia (determinismo) = {mf.mo_coeff[1,5]:.12f}")
print(f"\nA. degeneracion w(11) vs w(12) con conv apretada:")
print(f"   w(11) = {w[10]:.12e}")
print(f"   w(12) = {w[11]:.12e}")
print(f"   |dif| = {abs(w[10]-w[11]):.3e}   relativa = {abs(w[10]-w[11])/w[10]:.3e}")
print(f"   -> {'EXACTAMENTE DEGENERADOS' if abs(w[10]-w[11])/w[10] < 1e-6 else 'SEPARADOS'}")
print(f"\nB. conteos en el gauge de simetria (cruce -> multiplete completo):")
for k in ("90", "99", "99.9"):
    print(f"   {k:>5}% : {c[k][0]} -> {c[k][1]}")
print(f"\n   cum(10..20): " + " ".join(f"{cum[i]:.6f}" for i in range(9, 20)))
print(f"   w_max = {w[0]:.12f}")

print("\n" + "=" * 74)
print("C + D.  ¿QUE ES REALMENTE INVARIANTE DE GAUGE?")
print("=" * 74)
# espectro de Schmidt: valores singulares de la matriz CI (particion alpha|beta)
sv = np.linalg.svd(civec, compute_uv=False)
sq = np.sort(sv ** 2)[::-1]
cs, cums = counts(sq)
print(f"espectro de Schmidt: n90={cs['90'][0]}  n99={cs['99'][0]}  n99.9={cs['99.9'][0]}")

rng = np.random.default_rng(7)
shells = [(3, 4), (5, 6), (8, 9)]          # indices ACTIVOS de las tres capas pi
rows = []
for t in range(40):
    U = np.eye(NCAS)
    for a, b in shells:
        th = rng.uniform(0, 2 * np.pi)
        U[np.ix_([a, b], [a, b])] = [[np.cos(th), -np.sin(th)],
                                     [np.sin(th), np.cos(th)]]
    ci2 = fci.addons.transform_ci_for_orbital_rotation(civec, NCAS, NELECAS, U)
    w2 = np.sort((ci2 ** 2).sum(axis=1))[::-1]
    c2, cum2 = counts(w2)
    sv2 = np.sort(np.linalg.svd(ci2, compute_uv=False) ** 2)[::-1]
    cs2, _ = counts(sv2)
    rows.append((c2["90"][0], c2["90"][1], c2["99.9"][0], cum2[10], cum2[15],
                 cum2[19], w2[0], cs2["90"][0], cs2["99"][0], cs2["99.9"][0]))
A = np.array(rows)
print(f"\nsobre 40 rotaciones aleatorias dentro de las capas degeneradas:")
print(f"  n90 (cruce)          : valores {sorted(set(A[:,0].astype(int)))}")
print(f"  n90 (multiplete)     : valores {sorted(set(A[:,1].astype(int)))}")
print(f"  n99.9 (cruce)        : rango [{int(A[:,2].min())}, {int(A[:,2].max())}]")
print(f"  cum(11)              : dispersion {np.ptp(A[:,3]):.2e}   <- gauge-dependiente")
print(f"  cum(16)              : dispersion {np.ptp(A[:,4]):.2e}")
print(f"  cum(20)              : dispersion {np.ptp(A[:,5]):.2e}   media {A[:,5].mean():.9f}")
print(f"  w_max                : dispersion {np.ptp(A[:,6]):.2e}   media {A[:,6].mean():.9f}")
print(f"  Schmidt n90/n99/n999 : {sorted(set(A[:,7].astype(int)))} / "
      f"{sorted(set(A[:,8].astype(int)))} / {sorted(set(A[:,9].astype(int)))}")
print(f"\n  E_FCI es invariante por construccion (rotacion exacta del vector CI).")
