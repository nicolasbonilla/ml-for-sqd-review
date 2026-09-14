# -*- coding: utf-8 -*-
r"""Genera el conteo de C2 a 1.24 A que 3.1 afirma y que no tenia generador.

EL PAPER DICE: "in C2 at 1.24 A multiplet completion WIDENS the 90% range,
from 4-5 to 4-6" -- es decir, que la estabilizacion por completar el multiplete
que se observa en N2 NO es una regla general.

Se citaba sin generador; este fichero lo provee.

METODO. Se resuelve el FCI UNA vez y se transforma el VECTOR CI bajo cada rotacion
intra-capa con fci.addons.transform_ci_for_orbital_rotation, como hace
gauge_study/invariancia_general.py. Re-correr CASCI con orbitales rotados falla:
la simetria adaptada de pyscf rechaza orbitales que ya no llevan etiqueta de irrep.
"""
import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.fci import cistring

NCAS, NELECAS = 12, (4, 4)          # C2 (8e,12o) -> C(12,4) = 495 cadenas alpha
R_C2, NROT, SEMILLA = 1.24, 40, 0


def cuenta(w, thr=0.90, tol=1e-10):
    """cruce crudo del umbral, y con el multiplete degenerado completado."""
    o = np.sort(w)[::-1]
    n = int(np.searchsorted(np.cumsum(o), thr) + 1)
    m = n
    while m < len(o) and abs(o[m] - o[n - 1]) <= tol * max(1.0, abs(o[n - 1])):
        m += 1
    return n, m


mol = gto.M(atom="C 0 0 0; C 0 0 %.2f" % R_C2, basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
cas = mcscf.CASCI(mf, NCAS, NELECAS); cas.fcisolver.conv_tol = 1e-13
e0 = cas.kernel()[0]
C = np.asarray(cas.ci)
ncore = cas.ncore
eps = mf.mo_energy[ncore:ncore + NCAS]

capas, i = [], 0
while i < NCAS:
    j = i
    while j + 1 < NCAS and abs(eps[j + 1] - eps[i]) < 1e-8:
        j += 1
    if j > i:
        capas.append(list(range(i, j + 1)))
    i = j + 1

print("C2 R=%.2f A  CAS(8e,12o)  %d cadenas alpha" % (R_C2, cistring.num_strings(NCAS, 4)))
print("E_FCI = %.12f Ha" % e0)
print("capas degeneradas del espacio activo:", capas)

w0 = (C ** 2).sum(axis=1); w0 /= w0.sum()
n0, m0 = cuenta(w0)
print("gauge canonico: cruce crudo n=%d -> completado n=%d" % (n0, m0))

rng = np.random.default_rng(SEMILLA)
crudo, comp, dE = [], [], []
for k in range(NROT):
    U = np.eye(NCAS)
    for capa in capas:
        d = len(capa)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        U[np.ix_(capa, capa)] = Q
    C2 = fci.addons.transform_ci_for_orbital_rotation(C, NCAS, NELECAS, U)
    w = (np.asarray(C2) ** 2).sum(axis=1); w /= w.sum()
    a, b = cuenta(w)
    crudo.append(a); comp.append(b)
    dE.append(abs(np.vdot(C2.ravel(), C2.ravel()).real - 1.0))

crudo, comp = np.array(crudo), np.array(comp)
print()
print("sobre %d rotaciones intra-capa aleatorias:" % NROT)
print("  norma del vector CI conservada a %.2e" % max(dE))
print("  cruce CRUDO          : %d-%d   valores %s" % (crudo.min(), crudo.max(), sorted(set(crudo.tolist()))))
print("  multiplete COMPLETADO: %d-%d   valores %s" % (comp.min(), comp.max(), sorted(set(comp.tolist()))))
ac, am = crudo.max() - crudo.min(), comp.max() - comp.min()
print()
print("  => completar el multiplete %s el rango (anchura %d -> %d)"
      % ("ENSANCHA" if am > ac else ("ESTRECHA" if am < ac else "no cambia"), ac, am))
print("  EL PAPER AFIRMA: de 4-5 a 4-6, es decir ENSANCHA")
