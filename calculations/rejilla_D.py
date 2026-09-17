"""How the classical selector and the exact-weight ranking behave as D grows.

WHY THIS EXISTS. Section 5.1 sets the paper's own operative standard: not "reaches chemical
accuracy" but "lands inside the 1.3 mHa spread of the best classical solvers". Every
calculation the paper performs itself is then at D = 120, and at that dimension on stretched
N2 even the exact-weight top-D ranking leaves 9.71 mHa -- six times chemical accuracy and
seven and a half times the classical spread. A referee's objection follows immediately: the
design point is one at which no proposer, including a perfect one, can reach the bar the
paper set three sections earlier, so the orderings are being read off in a regime none of
them is fit for.

This script answers the objection with a measurement instead of an argument. It sweeps D and
reports, for each molecule, the dimension at which each arm enters the 1.3 mHa band.

WHAT IS AND IS NOT SWEPT. Only the two DETERMINISTIC arms:
  en_ci    the iterative Epstein-Nesbet selector of Fig. 7 (the paper's classical baseline)
  ranked   the top-D strings by exact FCI weight -- a ranking, not a sampler
Both are deterministic, so a single run is not an anecdote and no seeds are needed. The
sampling arms are deliberately NOT extended: a sampler's D-dependence is a different
quantity (it must collect D distinct strings, not take them) and extending it here would
repeat the confound this addition exists to expose.

THE INNER SOLVER, AND WHY IT CHANGED. An earlier version of this script diagonalized each
subspace with scipy's eigsh on a LinearOperator whose matvec embedded the D-string block
into the FULL dA x dA grid and called contract_2e on all of it. That is correct but its cost
per iteration does not depend on D at all, so the sweep cost about nine hours and N2 was cut
off at D = 180 -- leaving the paper's own flagship system without the crossing this script
exists to measure. pyscf.fci.selected_ci.kernel_fixed_space solves the identical problem --
the product subspace spanned by the chosen alpha-strings in both spin sectors -- contracting
only inside it. The change is an optimization, not a redefinition, and the script refuses to
write anything unless every previously deposited point reproduces (see COMPROBAR below).

One trap, recorded because it silently produces wrong answers rather than an error:
kernel_fixed_space addresses determinants by binary search and therefore requires the string
list to be SORTED. cistring.make_strings does not return strings in sorted order, so the CI
matrix has to be permuted back before the Epstein-Nesbet selector reads it. Without that the
selector scores the right numbers against the wrong strings.

    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci \
        python /w/calculations/rejilla_D.py
"""
import io
import json
import os
import time

import numpy as np
import pyscf
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.fci import cistring, direct_spin1, selected_ci

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

BANDA = 1.3      # mHa: la dispersion clasica que la seccion 5.1 declara como la vara
QUIMICA = 1.6    # mHa: exactitud quimica, para referencia
TOL_REPRO = 1e-3  # mHa: cuanto puede moverse un punto ya depositado antes de abortar

MOLS = {
    "h2o": dict(atom="O 0 0 0; H 0 0.98 0.76; H 0 -0.98 0.76", ncore=1, ncas=12,
                nelecas=(4, 4), dims=[30, 60, 120, 180, 240, 300]),
    # N2 llega hasta 400: la banda de 1.3 mHa se cruza dentro de la rejilla y el punto
    # siguiente la confirma. Con el solucionador correcto la cola cuesta minutos.
    "n2": dict(atom="N 0 0 0; N 0 0 2.0", ncore=2, ncas=12, nelecas=(5, 5),
               dims=[60, 120, 180, 240, 280, 300, 320, 400]),
}

# Los nueve puntos ya depositados. Si el cambio de solucionador mueve cualquiera de ellos
# mas de TOL_REPRO, este script no escribe nada: el deposito manda sobre la optimizacion.
COMPROBAR = {
    ("h2o", 30): (11.6528, 11.5163), ("h2o", 60): (3.2517, 3.1704),
    ("h2o", 120): (0.5634, 0.5753), ("h2o", 180): (0.1766, 0.1754),
    ("h2o", 240): (0.0495, 0.0505), ("h2o", 300): (0.0139, 0.0139),
    ("n2", 60): (21.6662, 23.6808), ("n2", 120): (9.1261, 9.7063),
    ("n2", 180): (4.6227, 4.8202),
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, os.pardir, "results", "rejilla_D.json")

_desvios = []


def corre(nombre, S):
    NC, NE = S["ncas"], S["nelecas"]
    a = NE[0]
    mol = gto.M(atom=S["atom"], basis="cc-pvdz", symmetry=True, verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
    cas = mcscf.CASCI(mf, NC, NE); cas.ncore = S["ncore"]
    H1, EC = cas.get_h1cas()
    H2 = ao2mo.restore(1, cas.get_h2cas(), NC)
    sA = cistring.make_strings(range(NC), a)
    dA = len(sA)
    hfi = int(np.where(sA == (1 << a) - 1)[0][0])
    H2e = direct_spin1.absorb_h1e(H1, H2, NC, NE, 0.5)
    hd = direct_spin1.make_hdiag(H1, H2, NC, NE).reshape(dA, dA)

    myci = selected_ci.SCI(); myci.verbose = 0
    solver = pyscf.fci.direct_spin1.FCI(); solver.conv_tol = 1e-13
    eF, cV = solver.kernel(H1, H2, NC, NE, ecore=EC)
    cV = cV.reshape(dA, dA)
    wT = (cV ** 2).sum(1)
    orden = np.argsort(wT)[::-1]
    log("%s: %d cadenas alpha, E_FCI=%.9f Ha, gauge=%s" % (nombre, dA, eF, mol.groupname))

    def fundamental(idx):
        """Estado fundamental del subespacio producto idx x idx.

        Devuelve (energia electronica, C en el orden de idx, idx). kernel_fixed_space
        direcciona los determinantes por busqueda binaria y exige la lista ORDENADA, de modo
        que hay que ordenar las cadenas y deshacer la permutacion sobre C antes de
        devolverla: el selector EN lee C para puntuar cadenas y una permutacion silenciosa
        le haria puntuar los numeros correctos contra las cadenas equivocadas.
        """
        idx = np.asarray(sorted(set(int(x) for x in idx)))
        n = len(idx)
        if n == 1:
            full = np.zeros((dA, dA)); full[idx[0], idx[0]] = 1.0
            e = direct_spin1.contract_2e(H2e, full, NC, NE).reshape(dA, dA)[idx[0], idx[0]]
            return float(e), np.array([[1.0]]), idx

        strs = sA[idx]
        perm = np.argsort(strs)
        strs_ord = np.ascontiguousarray(strs[perm], dtype=sA.dtype)
        e, c = selected_ci.kernel_fixed_space(myci, H1, H2, NC, NE,
                                              ci_strs=(strs_ord, strs_ord),
                                              ecore=EC, tol=1e-13, max_cycle=400)
        c = np.asarray(c).reshape(n, n)
        inv = np.argsort(perm)
        return float(e) - EC, c[np.ix_(inv, inv)], idx

    def err(idx):
        return (fundamental(idx)[0] + EC - eF) * 1000

    filas = []
    for D in S["dims"]:
        if D >= dA:
            continue
        ta = time.time()
        # el selector EN iterativo, como en calculations/hci_baseline.py
        A = {hfi}
        lote = max(12, D // 8)
        while len(A) < D:
            e_el, cc, idx = fundamental(A)
            full = np.zeros((dA, dA)); full[np.ix_(idx, idx)] = cc
            Hc2 = direct_spin1.contract_2e(H2e, full, NC, NE).reshape(dA, dA)
            gap = np.where(np.abs(e_el - hd) < 1e-6, 1e-6, e_el - hd)
            sc = ((Hc2 ** 2) / (gap ** 2)).sum(1)
            sc[list(A)] = -np.inf
            A.update(np.argsort(sc)[::-1][:min(lote, D - len(A))].tolist())
        e_en = err(A)
        e_rk = err(orden[:D])
        filas.append({"D": D, "en_ci_mHa": round(e_en, 4), "ranked_mHa": round(e_rk, 4)})

        marca = ""
        if (nombre, D) in COMPROBAR:
            dep_en, dep_rk = COMPROBAR[(nombre, D)]
            d1, d2 = abs(e_en - dep_en), abs(e_rk - dep_rk)
            _desvios.append((nombre, D, d1, d2))
            marca = "  [deposito %.4f/%.4f, delta %.1e/%.1e]" % (dep_en, dep_rk, d1, d2)
        log("   D=%-4d  EN-CI %8.3f   ranked |c|^2 %8.3f   (banda %.1f, %.0fs)%s"
            % (D, e_en, e_rk, BANDA, time.time() - ta, marca))

    def entra(clave):
        for f in filas:
            if f[clave] <= BANDA:
                return f["D"]
        return None

    return {"atom": S["atom"], "ncas": NC, "nelecas": list(NE), "E_FCI_Ha": float(eF),
            "n_alpha_strings": dA, "banda_mHa": BANDA, "filas": filas,
            "D_entra_en_banda": {"en_ci": entra("en_ci_mHa"), "ranked": entra("ranked_mHa")}}


def main():
    res = {"_nota": ("Barrido de la dimension del subespacio. Responde a la objecion de que "
                     "todo el trabajo propio del paper corre a D=120, donde ni el mejor "
                     "seleccionador posible alcanza la vara de 1.3 mHa que el propio paper "
                     "fija en la seccion 5.1. Solo brazos DETERMINISTAS: no hay semillas "
                     "porque no hay azar."),
           "_solucionador": ("pyscf.fci.selected_ci.kernel_fixed_space. Una version anterior "
                             "usaba scipy.eigsh sobre un LinearOperator que incrustaba el "
                             "bloque de D cadenas en la rejilla COMPLETA dA x dA, de modo que "
                             "el coste por iteracion no dependia de D; el barrido costaba unas "
                             "nueve horas y N2 se cortaba en D=180, justo antes de la banda. "
                             "El cambio es una optimizacion, no una redefinicion: los nueve "
                             "puntos de la version anterior se recomputan y el script aborta "
                             "sin escribir si alguno se mueve mas de 1e-3 mHa."),
           "gauge": "D-infinity-h adaptado por simetria, OMP_NUM_THREADS=1",
           "mols": {}}
    for nombre, S in MOLS.items():
        res["mols"][nombre] = corre(nombre, S)

    print()
    print("=" * 74)
    peor = max([max(d1, d2) for _, _, d1, d2 in _desvios] or [0.0])
    print("  puntos ya depositados recomputados: %d" % len(_desvios))
    print("  peor desviacion contra el deposito: %.2e mHa  (tolerancia %.0e)" % (peor, TOL_REPRO))
    if peor > TOL_REPRO:
        print()
        for n, D, d1, d2 in _desvios:
            if max(d1, d2) > TOL_REPRO:
                print("    %s D=%-4d  delta EN-CI %.2e  delta ranked %.2e" % (n, D, d1, d2))
        print("  EL CAMBIO DE SOLUCIONADOR NO ES NEUTRAL. No se escribe nada.")
        raise SystemExit(1)
    print()
    for k, v in res["mols"].items():
        e = v["D_entra_en_banda"]
        print("  %-4s entra en la banda de %.1f mHa:  EN-CI en D=%s   ranking exacto en D=%s"
              % (k, BANDA, e["en_ci"] or ">rejilla", e["ranked"] or ">rejilla"))
        d = e["en_ci"]
        if d:
            print("       %d de %d cadenas (%.0f%%), %d de %d determinantes (%.0f%%)"
                  % (d, v["n_alpha_strings"], 100.0 * d / v["n_alpha_strings"],
                     d * d, v["n_alpha_strings"] ** 2,
                     100.0 * d * d / v["n_alpha_strings"] ** 2))
    print("=" * 74)

    io.open(DEST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    log("WROTE %s" % os.path.normpath(DEST))


if __name__ == "__main__":
    main()
