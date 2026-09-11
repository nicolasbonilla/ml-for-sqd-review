# -*- coding: utf-8 -*-
"""LA PRUEBA DECISIVA.

Hasta ahora solo rote DENTRO de capas degeneradas. La pregunta real es mas fuerte:
el espectro de Schmidt alpha|beta, ¿es invariante bajo CUALQUIER rotacion orbital
independiente del espin -- incluidos orbitales naturales, localizados, o una
rotacion general de SO(12)?

Si lo es, entonces el coste de 'entanglement forging' a traves del corte de espin
NO se puede reducir con optimizacion orbital independiente del espin. Eso toca
directamente el roadmap de Eddins et al. (PRX Quantum 3, 010309), que propone
justamente 'adaptive bisection based on orbital optimization' como trabajo futuro.
"""
import numpy as np
from pyscf import ao2mo, fci, gto, lo, mcscf, scf
from scipy.stats import ortho_group

mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()
cas = mcscf.CASCI(mf, 12, (5, 5))
h1, ec = cas.get_h1cas()
h2 = ao2mo.restore(1, cas.get_h2cas(), 12)
s = fci.direct_spin1.FCI()
s.conv_tol = 1e-13
e, C = s.kernel(h1, h2, 12, (5, 5))
print(f"E_FCI = {e + ec:.12f}\n")


def analiza(M):
    sv = np.sort(np.linalg.svd(M, compute_uv=False) ** 2)[::-1]
    cs = np.cumsum(sv)
    w = np.sort((M ** 2).sum(1))[::-1]
    cw = np.cumsum(w)
    n_s = [int(np.searchsorted(cs, p) + 1) for p in (0.90, 0.99, 0.999)]
    n_w = [int(np.searchsorted(cw, p) + 1) for p in (0.90, 0.99, 0.999)]
    return sv, n_s, n_w


sv0, ns0, nw0 = analiza(C)
print(f"REFERENCIA   Schmidt n90/n99/n99.9 = {ns0}   conteo de cadenas = {nw0}")
print(f"             5 mayores valores de Schmidt: {np.round(sv0[:5], 10)}\n")

rng = np.random.default_rng(2026)
print("rotacion orbital GENERAL de SO(12), identica para alpha y beta:")
print("   #   max|dif| espectro      Schmidt n90/99/99.9    cadenas n90/99/99.9")
peor = 0.0
for t in range(8):
    U = ortho_group.rvs(12, random_state=int(rng.integers(1 << 31)))
    C2 = fci.addons.transform_ci_for_orbital_rotation(C, 12, (5, 5), U)
    sv, ns, nw = analiza(C2)
    d = float(np.max(np.abs(sv - sv0)))
    peor = max(peor, d)
    print(f"   {t}     {d:.3e}              {str(ns):18s}   {nw}")

print(f"\n   peor discrepancia del espectro de Schmidt: {peor:.3e}")
print(f"   => {'INVARIANTE bajo TODA rotacion orbital' if peor < 1e-9 else 'NO invariante'}")

print("\ncasos con significado fisico:")
dm1 = s.make_rdm1(C, 12, (5, 5))
occ, natorb = np.linalg.eigh(dm1)
natorb = natorb[:, ::-1]
C2 = fci.addons.transform_ci_for_orbital_rotation(C, 12, (5, 5), natorb)
sv, ns, nw = analiza(C2)
print(f"   orbitales NATURALES : max|dif| = {np.max(np.abs(sv - sv0)):.3e}"
      f"   Schmidt {ns}   cadenas {nw}")

try:
    mo_act = mf.mo_coeff[:, 2:14]
    locorb = lo.Boys(mol, mo_act).kernel()
    U = mo_act.T @ mf.get_ovlp() @ locorb
    C2 = fci.addons.transform_ci_for_orbital_rotation(C, 12, (5, 5), U)
    sv, ns, nw = analiza(C2)
    print(f"   orbitales de BOYS   : max|dif| = {np.max(np.abs(sv - sv0)):.3e}"
          f"   Schmidt {ns}   cadenas {nw}")
except Exception as ex:
    print(f"   Boys fallo: {type(ex).__name__}: {ex}")
