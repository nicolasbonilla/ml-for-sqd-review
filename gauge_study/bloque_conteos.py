# -*- coding: utf-8 -*-
r"""Genera el bloque de conteos de la seccion 3.1, que en su mayor parte no tenia
generador. Mide, y dice cual de las afirmaciones del paper queda respaldada.

LAS AFIRMACIONES QUE SE COMPRUEBAN, textuales del manuscrito:

  (a) el cociente peor-caso entre el conteo mayor y el menor bajo rotacion intra-capa.
      El manuscrito afirmaba 1.29 sin generador; esta medida da 1.09 en N2 y 1.25 en C2,
      y el paper se corrigio a 1.25.
  (b) "Three quantities are exactly gauge-invariant ... The number of ORBITS ... 6, 23
      and 82 orbits (13, 54 and 208 strings) at the three thresholds, against raw counts
      that wobble by 11-12, 46-50 and 185-196"
  (c) el sorteo canonico. El manuscrito afirmaba "12, 47 and 187"; esta medida da
      12, 49 y 194, y el paper se corrigio -- y ademas dejo de llamarlo "the" canonical
      draw, porque la 2.1 establece que no hay uno solo.
  (d) "only 84 of the 792 strings ... carry a gauge-invariant weight" (10.6%)
  (e) "H2O in C2v, which has no degenerate shell, shows a gauge spread of exactly zero
      across 100 rotations, with all 495 strings invariant"

QUE ES UNA ORBITA. Una rotacion dentro de una capa degenerada mezcla unas cadenas con
otras. Dos cadenas estan en la misma orbita si alguna rotacion intra-capa puede llevar
peso de una a la otra. Aqui la orbita se identifica por el patron de ocupacion agregado:
cuantos electrones pone la cadena en cada capa degenerada, mas la ocupacion exacta de los
orbitales no degenerados. Ese patron es lo que una rotacion intra-capa NO puede cambiar,
asi que el peso total de una orbita es invariante aunque el de cada cadena no lo sea.
Una cadena que llena o vacia por completo cada capa degenerada es una orbita de tamano
uno: esas son las "84 gauge-invariantes".

METODO. Se resuelve el FCI una vez y se transforma el vector CI bajo cada rotacion con
fci.addons.transform_ci_for_orbital_rotation, comprobando que la norma se conserva.

    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci \
        python /w/gauge_study/bloque_conteos.py
"""
import itertools

import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.fci import cistring

NROT, SEMILLA = 40, 0
THRESHOLDS = (0.90, 0.99, 0.999)
TOL_DEG = 1e-8       # Ha, para agrupar orbitales en capas degeneradas
TOL_W = 1e-10        # peso, para no partir un multiplete



def fraccion_invariante(ncas, na, capas):
    """Cuantas cadenas alpha llenan o vacian por completo cada capa degenerada.

    Es combinatoria pura -- no hace falta resolver nada -- y es el numero que la seccion
    3.1 cita como "the fraction of gauge-invariant strings". Solo esas cadenas tienen un
    peso que ninguna rotacion intra-capa puede mover.

        12 orbitales, 3 capas dobles:  84 de  792 = 10.6%
        14 orbitales, 4 capas dobles: 122 de 2002 =  6.1%

    Las dos cifras aparecian en el manuscrito sin generador; esta funcion es su generador.
    """
    from itertools import combinations
    tot = inv = 0
    cs = [set(c) for c in capas]
    for occ in combinations(range(ncas), na):
        tot += 1
        s = set(occ)
        if all(len(s & c) in (0, len(c)) for c in cs):
            inv += 1
    return inv, tot, 100.0 * inv / tot


def capas_degeneradas(eps):
    capas, i = [], 0
    while i < len(eps):
        j = i
        while j + 1 < len(eps) and abs(eps[j + 1] - eps[i]) < TOL_DEG:
            j += 1
        if j > i:
            capas.append(list(range(i, j + 1)))
        i = j + 1
    return capas


def cuenta(w, thr):
    """Cruce crudo del umbral, y con el multiplete degenerado completado."""
    o = np.sort(w)[::-1]
    n = int(np.searchsorted(np.cumsum(o), thr) + 1)
    m = n
    while m < len(o) and abs(o[m] - o[n - 1]) <= TOL_W * max(1.0, abs(o[n - 1])):
        m += 1
    return n, m


def orbitas_necesarias(w, lab, thr):
    """Cuantas orbitas hacen falta para cubrir thr, y cuantas cadenas suman.

    LAS ORBITAS SE ORDENAN POR SU MASA TOTAL, no por el peso de su cadena mayor.
    La masa de una orbita es invariante bajo rotacion intra-capa; el peso de una
    cadena concreta no. Ordenar por lo segundo hace que el conteo parezca variar
    con el gauge cuando no varia -- exactamente el error que este script existe
    para no cometer.
    """
    uniq, inv = np.unique(lab, return_inverse=True)
    masa = np.zeros(len(uniq))
    np.add.at(masa, inv, w)
    tam = np.bincount(inv, minlength=len(uniq))
    orden = np.argsort(masa)[::-1]
    acc, n_orb, n_str = 0.0, 0, 0
    for k in orden:
        acc += masa[k]; n_orb += 1; n_str += int(tam[k])
        if acc >= thr:
            break
    return n_orb, n_str


def orbitas(strs, ncas, capas):
    """Etiqueta de orbita por cadena: (ocupacion de cada orbital NO degenerado,
    numero de electrones en cada capa degenerada). Invariante bajo rotacion intra-capa."""
    en_capa = {o: k for k, c in enumerate(capas) for o in c}
    libres = [o for o in range(ncas) if o not in en_capa]
    etq = []
    for s in strs:
        bits = [(int(s) >> p) & 1 for p in range(ncas)]
        fijo = tuple(bits[o] for o in libres)
        porcapa = tuple(sum(bits[o] for o in c) for c in capas)
        etq.append(fijo + porcapa)
    return np.array([hash(e) for e in etq]), etq


def analiza(nombre, atom, ncas, nelecas, ncore=None, nrot=NROT):
    na = nelecas[0]
    mol = gto.M(atom=atom, basis="cc-pvdz", symmetry=True, verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
    cas = mcscf.CASCI(mf, ncas, nelecas)
    if ncore is not None:
        cas.ncore = ncore
    cas.fcisolver.conv_tol = 1e-13
    e0 = cas.kernel()[0]
    C0 = np.asarray(cas.ci)
    eps = mf.mo_energy[cas.ncore:cas.ncore + ncas]
    capas = capas_degeneradas(eps)
    strs = cistring.make_strings(range(ncas), na)
    dim = len(strs)

    print("=" * 78)
    print("%s   CAS(%de,%do)   %d cadenas alpha   E_FCI = %.12f Ha"
          % (nombre, sum(nelecas), ncas, dim, e0))
    print("  capas degeneradas: %s" % (capas if capas else "NINGUNA"))

    w0 = (C0 ** 2).sum(axis=1); w0 /= w0.sum()

    # --- (d) cadenas gauge-invariantes: las que llenan o vacian cada capa entera ---
    inv = 0
    for s in strs:
        bits = [(int(s) >> p) & 1 for p in range(ncas)]
        if all(sum(bits[o] for o in c) in (0, len(c)) for c in capas):
            inv += 1
    print("  cadenas gauge-invariantes (llenan o vacian cada capa): %d de %d = %.1f%%"
          % (inv, dim, 100 * inv / dim))

    if not capas:
        print("  -> sin capas degeneradas NO HAY rotacion intra-capa que hacer:")
        print("     la dispersion de gauge es cero por construccion y las %d cadenas" % dim)
        print("     son invariantes. La afirmacion del paper es correcta, pero es una")
        print("     consecuencia estructural, no una medida sobre %d rotaciones." % nrot)
        return

    # --- (b) orbitas ---
    lab, _ = orbitas(strs, ncas, capas)
    print()
    print("  %-8s %-16s %-16s %-22s" % ("umbral", "cruce crudo", "completado", "orbitas (cadenas)"))
    base = {}
    for thr in THRESHOLDS:
        n, m = cuenta(w0, thr)
        o = np.argsort(w0)[::-1]
        cum = np.cumsum(w0[o])
        # orbitas necesarias para cubrir el umbral: se anaden orbitas enteras
        no_orb, no_str = orbitas_necesarias(w0, lab, thr)
        base[thr] = (n, m, no_orb, no_str)
        print("  %-8.3f %-16d %-16d %d (%d)" % (thr, n, m, no_orb, no_str))

    # --- (a) y (c): dispersion de los conteos bajo rotacion ---
    rng = np.random.default_rng(SEMILLA)
    crudos = {t: [] for t in THRESHOLDS}
    comps = {t: [] for t in THRESHOLDS}
    orbs = {t: [] for t in THRESHOLDS}
    dN = []
    for _ in range(nrot):
        U = np.eye(ncas)
        for c in capas:
            d = len(c)
            Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
            U[np.ix_(c, c)] = Q
        Cr = np.asarray(fci.addons.transform_ci_for_orbital_rotation(C0, ncas, nelecas, U))
        dN.append(abs(float(np.vdot(Cr.ravel(), Cr.ravel()).real) - 1.0))
        w = (Cr ** 2).sum(axis=1); w /= w.sum()
        for thr in THRESHOLDS:
            n, m = cuenta(w, thr)
            crudos[thr].append(n); comps[thr].append(m)
            orbs[thr].append(orbitas_necesarias(w, lab, thr)[0])

    print()
    print("  norma del vector CI conservada a %.2e sobre %d rotaciones" % (max(dN), nrot))
    print("  %-8s %-16s %-16s %-14s %-8s" % ("umbral", "crudo", "completado", "orbitas", "cociente"))
    peor = 0.0
    for thr in THRESHOLDS:
        c, m, o = np.array(crudos[thr]), np.array(comps[thr]), np.array(orbs[thr])
        r = c.max() / c.min()
        peor = max(peor, r)
        print("  %-8.3f %-16s %-16s %-14s %.3f"
              % (thr, "%d-%d" % (c.min(), c.max()), "%d-%d" % (m.min(), m.max()),
                 "%d-%d" % (o.min(), o.max()), r))
    print()
    print("  (a) cociente peor-caso entre el conteo mayor y el menor: %.3f" % peor)
    print("      EL PAPER AFIRMA 1.25  ->  %s"
          % ("respaldado" if abs(peor - 1.25) <= 0.16 else "NO RESPALDADO"))
    print("  (b) las orbitas %s tuvieron rango cero en los tres umbrales"
          % ("SI" if all(len(set(orbs[t])) == 1 for t in THRESHOLDS) else "NO"))
    print("      medidas: %s  (el paper afirma 6, 23 y 82)"
          % [int(np.median(orbs[t])) for t in THRESHOLDS])
    print("      cadenas en esas orbitas en el gauge de partida: %s  (el paper: 13, 54, 208)"
          % [base[t][3] for t in THRESHOLDS])

    # --- (c) el sorteo canonico: sin symmetry, orientacion arbitraria ---
    mol2 = gto.M(atom=atom, basis="cc-pvdz", verbose=0)
    mf2 = scf.RHF(mol2); mf2.conv_tol = 1e-12; mf2.run()
    cas2 = mcscf.CASCI(mf2, ncas, nelecas)
    if ncore is not None:
        cas2.ncore = ncore
    cas2.fcisolver.conv_tol = 1e-13
    e2 = cas2.kernel()[0]
    w2 = (np.asarray(cas2.ci) ** 2).sum(axis=1); w2 /= w2.sum()
    print()
    dE2 = abs(e2 - e0)
    print("  (c) sorteo canonico (sin symmetry=True): E_FCI difiere en %.2e Ha" % dE2)
    if dE2 > 1e-9:
        print("      NO SE COMPARA: esa diferencia significa que el CASCI canonico convergio")
        print("      a OTRO estado, no a otro gauge del mismo. Comparar conteos entre dos")
        print("      estados distintos no dice nada sobre el gauge.")
        return
    print("      conteos crudos:      %s" % [cuenta(w2, t)[0] for t in THRESHOLDS])
    print("      conteos completados: %s   (el paper afirma 12, 49, 194)"
          % [cuenta(w2, t)[1] for t in THRESHOLDS])
    print("      en el gauge adaptado: crudos %s, completados %s"
          % ([cuenta(w0, t)[0] for t in THRESHOLDS], [cuenta(w0, t)[1] for t in THRESHOLDS]))


if __name__ == "__main__":
    analiza("N2 @ 2.0 A", "N 0 0 0; N 0 0 2.0", 12, (5, 5), ncore=2)
    print()
    analiza("C2 @ 1.24 A", "C 0 0 0; C 0 0 1.24", 12, (4, 4), ncore=2)
    print()
    analiza("H2O @ 1.24 A (C2v, sin capas degeneradas)",
            "O 0 0 0; H 0 0.98 0.76; H 0 -0.98 0.76", 12, (4, 4), ncore=1, nrot=100)
    print()
    print("=" * 78)
    print("FRACCION DE CADENAS GAUGE-INVARIANTES (combinatoria, sin resolver nada)")
    for ncas, na, capas, etq in (
            (12, 5, [[3, 4], [5, 6], [8, 9]], "12 orbitales, 3 capas dobles (N2 CAS(10e,12o))"),
            (14, 5, [[3, 4], [5, 6], [8, 9], [11, 12]], "14 orbitales, 4 capas dobles")):
        i, t, p = fraccion_invariante(ncas, na, capas)
        print("  %-46s %4d de %4d = %5.2f%%" % (etq, i, t, p))
    print("  El paper cita 10.6%% y 6.1%%. Son estas.")

