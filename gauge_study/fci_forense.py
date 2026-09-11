# -*- coding: utf-8 -*-
"""FORENSE DEL COUPON-COLLECTOR — P1, N2 CAS(10e,12o).

Tres preguntas, en orden:

 1. CANONICA. Con la receta EXACTA que declara el paper y que usan los ocho scripts
    del repositorio, ¿cuanto vale E_FCI y cuantas cadenas llevan el 90% del peso?
    Se imprime todo a precision completa y se compara contra los .dat publicados
    (11 / cum@11=0.90247) y contra el recompute de julio (12 / cum@11=0.8936).

 2. GAUGE. w_alpha NO es invariante bajo rotaciones dentro de una capa degenerada:
    la rotacion deja E_FCI intacta pero redistribuye el peso entre las cadenas.
    Si el conteo del 90% cambia al rotar los pi degenerados, entonces "11 vs 12"
    no es un error de nadie -- es una cantidad mal definida, y eso hay que decirlo
    en el paper. Esta es la hipotesis que explicaria las dos corridas de una vez.

 3. VARIANTES. Si el gauge no lo explica, se prueban los setups que si podrian:
    ncore distinto, orbitales naturales en vez de canonicos, CASSCF en vez de CASCI.

No escribe nada fuera de su propio JSON. No toca el paper.
"""
import json
import os

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

OUT = "/w/fci_forense_resultados.json"
R = {}

# ---------------------------------------------------------------------------
# La receta declarada. Identica en generalize.py, mve_backbone.py, coupon_fig.py,
# build_notebook.py y los cuatro generadores recuperados.
# ---------------------------------------------------------------------------
NCAS, NELECAS = 12, (5, 5)
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
R["e_rhf"] = float(mf.e_tot)
R["nao"] = int(mol.nao)
print(f"RHF        E = {mf.e_tot:.9f} Ha   ({mol.nao} funciones de base)")


def weights_and_cum(mo, tag, ncore=None):
    """CASCI en los orbitales dados -> (E_FCI, pesos alpha ordenados, acumulada)."""
    cas = mcscf.CASCI(mf, NCAS, NELECAS)
    if ncore is not None:
        cas.ncore = ncore
    cas.mo_coeff = mo
    h1, ecore = cas.get_h1cas()
    h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
    e, civec = fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS)
    efci = float(e + ecore)
    w = np.sort((civec ** 2).sum(axis=1))[::-1]     # marginal por cadena alpha
    return efci, w, np.cumsum(w), cas.ncore


def thresholds(w, cum):
    """Cruce simple y multiplete completo, para 90 / 99 / 99.9%."""
    out = {}
    for thr, lab in ((0.90, "90"), (0.99, "99"), (0.999, "99.9")):
        idx = int(np.searchsorted(cum, thr) + 1)     # 1-indexado
        k = idx
        # completar el multiplete: avanzar mientras el peso siga siendo el mismo
        while k < len(w) and abs(w[k] - w[k - 1]) <= 1e-12 * max(1.0, abs(w[k - 1])):
            k += 1
        out[lab] = dict(cross=idx, complete=k,
                        cum_cross=float(cum[idx - 1]), cum_complete=float(cum[k - 1]),
                        pct_complete=round(100.0 * k / len(w), 3))
    return out


# ===========================================================================
# 1. CANONICA
# ===========================================================================
print("\n" + "=" * 70 + "\n1. RECETA CANONICA DECLARADA\n" + "=" * 70)
efci, w, cum, ncore = weights_and_cum(mf.mo_coeff, "canonica")
R["canonical"] = dict(e_fci=efci, ncore=int(ncore), n_strings=int(len(w)),
                      w_sum=float(w.sum()),
                      w=[float(x) for x in w[:15]],
                      cum={str(n): float(cum[n - 1]) for n in range(8, 16)},
                      thresholds=thresholds(w, cum))
print(f"E_FCI      = {efci:.9f} Ha     (ncore={ncore}, {len(w)} cadenas alpha)")
print(f"  registro de la corrida de 12 cadenas: -108.808042 Ha")
print(f"\n  n :        w(n)          cum(n)")
for n in range(9, 15):
    print(f" {n:2d} : {w[n-1]:.6e}   {cum[n-1]:.6f}")
t = R["canonical"]["thresholds"]
print()
for lab in ("90", "99", "99.9"):
    d = t[lab]
    print(f"  {lab:>4}% : cruce n={d['cross']} -> multiplete completo n={d['complete']}"
          f"  (cum={d['cum_complete']:.5f}, {d['pct_complete']}%)")
print(f"\n  .dat de entonces: 11 / cum@11=0.90247 (sorteo canonico, ya sustituido)")
print(f"  gauge D-inf-h : 12 / cum@11=0.90565  <- el fichero actual")

# ===========================================================================
# 2. DEPENDENCIA DE GAUGE  (la hipotesis)
# ===========================================================================
print("\n" + "=" * 70 + "\n2. ¿ES INVARIANTE DE GAUGE EL CONTEO?\n" + "=" * 70)
occ_e = mf.mo_energy
shells = []
i = ncore
while i < ncore + NCAS:
    j = i
    while j + 1 < ncore + NCAS and abs(occ_e[j + 1] - occ_e[i]) < 1e-6:
        j += 1
    if j > i:
        shells.append(list(range(i, j + 1)))
    i = j + 1
R["degenerate_shells"] = [[int(x) for x in s] for s in shells]
print(f"capas degeneradas dentro del espacio activo (indices MO): {shells}")
for s in shells:
    print(f"   {s} -> energias {[round(float(occ_e[k]), 8) for k in s]}")

counts90, counts999, efcis = [], [], []
rng = np.random.default_rng(20260909)
for trial in range(12):
    mo = mf.mo_coeff.copy()
    for s in shells:                       # rotacion aleatoria dentro de cada capa
        th = rng.uniform(0, 2 * np.pi)
        c, sn = np.cos(th), np.sin(th)
        a, b = s[0], s[1]
        mo[:, [a, b]] = np.column_stack([c * mo[:, a] - sn * mo[:, b],
                                         sn * mo[:, a] + c * mo[:, b]])
    e2, w2, cum2, _ = weights_and_cum(mo, f"gauge{trial}")
    th2 = thresholds(w2, cum2)
    counts90.append(th2["90"]["complete"])
    counts999.append(th2["99.9"]["complete"])
    efcis.append(e2)
    print(f"  rotacion {trial:2d}: E_FCI={e2:.9f}   90% -> {th2['90']['complete']:3d} cadenas"
          f"   (cruce {th2['90']['cross']})   99.9% -> {th2['99.9']['complete']}")

R["gauge"] = dict(counts90=counts90, counts999=counts999,
                  e_fci_spread=float(max(efcis) - min(efcis)),
                  distinct90=sorted(set(counts90)))
print(f"\n  E_FCI invariante? dispersion = {max(efcis)-min(efcis):.2e} Ha")
print(f"  conteos del 90% observados: {sorted(set(counts90))}")
print("  VEREDICTO GAUGE:", "NO INVARIANTE -- el conteo depende del gauge"
      if len(set(counts90)) > 1 else "invariante en estas rotaciones")

# ===========================================================================
# 3. VARIANTES DE SETUP
# ===========================================================================
print("\n" + "=" * 70 + "\n3. VARIANTES QUE PODRIAN EXPLICAR 0.8936\n" + "=" * 70)
R["variants"] = {}
try:
    from pyscf import mp
    nat = mp.MP2(mf).run()
    _, natorb = nat.make_natorbs() if hasattr(nat, "make_natorbs") else (None, None)
except Exception:
    natorb = None

variants = {}
if natorb is not None:
    variants["mp2_natural_orbitals"] = (natorb, None)
variants["ncore_1"] = (mf.mo_coeff, 1)
variants["ncore_3"] = (mf.mo_coeff, 3)

for name, (mo, nc) in variants.items():
    try:
        e3, w3, cum3, nc3 = weights_and_cum(mo, name, ncore=nc)
        th3 = thresholds(w3, cum3)
        R["variants"][name] = dict(e_fci=e3, ncore=int(nc3),
                                   cum11=float(cum3[10]), cum12=float(cum3[11]),
                                   n90=th3["90"]["complete"])
        print(f"  {name:24s} E_FCI={e3:.6f}  cum@11={cum3[10]:.5f}  cum@12={cum3[11]:.5f}"
              f"  -> 90% en {th3['90']['complete']}")
    except Exception as ex:
        R["variants"][name] = dict(error=f"{type(ex).__name__}: {ex}")
        print(f"  {name:24s} FALLO: {type(ex).__name__}: {ex}")

with open(OUT, "w") as fh:
    json.dump(R, fh, indent=2)
print(f"\nescrito {OUT}")
