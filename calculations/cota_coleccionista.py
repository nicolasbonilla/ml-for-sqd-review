"""How loose the coupon-collector bound is on this system -- a claim that had no generator.

WHAT SECTION 3.1 CLAIMS. The expected number of draws to collect every string in a target
set T is bounded above by the coupon-collector sum, "a loose bound exactly where the
argument invokes it: evaluated on this system's exact weights it overstates the true
expectation by approximately 2.7x for the 90% target set and by approximately 17x for the
99.9% set".

WHY THIS SCRIPT EXISTS. Nothing in the deposit estimated a true collection time.
calculations/verify_coupon.py computes string counts only. The two factors were printed
without a generator, in the section whose own post-audit paragraph claims to have swept for
exactly that.

WHAT IS COMPUTED.
  BOUND   the leading inclusion-exclusion term, sum over the target set of 1/p_i. This is
          the quantity the manuscript calls "the coupon-collector sum". It is an upper
          bound on the expected time to collect T because it charges each coupon its own
          waiting time as though the others were never being collected in parallel.
  TRUTH   the expectation itself, by direct simulation: draw from the full 792-string
          distribution until every member of T has been seen, repeat NREP times, average.
  RATIO   bound / truth -- the overstatement factor the manuscript quotes.

The target set at each threshold is the smallest set of largest-weight strings whose weight
sums to at least the threshold, taken from the deposited results/coupon_w.dat, which is in
the declared D-infinity-h symmetry-adapted gauge and carries its recipe in its header.

No PySCF and no Docker: this reads a deposited file and draws from it.

    python calculations/cota_coleccionista.py
"""
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
DEST = os.path.join(ROOT, "results", "cota_coleccionista.json")

THRESHOLDS = (0.90, 0.99, 0.999)
NREP = 4000
SEMILLA = 0
CAP = 40_000_000     # tope de tiradas por repeticion, para no colgarse en la cola


def pesos():
    rows = [l.split() for l in io.open(os.path.join(ROOT, "results", "coupon_w.dat"),
                                       encoding="utf-8")
            if l.strip() and not l.lstrip().startswith("%")]
    w = np.array([float(r[1]) for r in rows[1:]])
    return w / w.sum()


def objetivo(w, thr):
    o = np.argsort(w)[::-1]
    k = int(np.searchsorted(np.cumsum(w[o]), thr)) + 1
    return o[:k]


def verdad(w, tgt, rng, nrep=NREP):
    """Tiempo esperado de coleccion, por simulacion directa.

    Se muestrea a bloques y se mira cuando el ULTIMO miembro del objetivo aparece por
    primera vez. Es exacto salvo error de Monte Carlo, y el error se reporta.
    """
    tset = set(int(x) for x in tgt)
    n = len(tset)
    tiempos = []
    for _ in range(nrep):
        vistos = set()
        t = 0
        bloque = max(4 * n, 512)
        while len(vistos) < n and t < CAP:
            d = rng.choice(len(w), size=bloque, p=w)
            for j, x in enumerate(d):
                xi = int(x)
                if xi in tset and xi not in vistos:
                    vistos.add(xi)
                    if len(vistos) == n:
                        t += j + 1
                        break
            else:
                t += bloque
                bloque = min(bloque * 2, 1 << 20)
                continue
            break
        tiempos.append(t)
    a = np.array(tiempos, dtype=float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def main():
    w = pesos()
    rng = np.random.default_rng(SEMILLA)
    print("  %d cadenas alpha, de results/coupon_w.dat (gauge adaptado por simetria)" % len(w))
    print("  %d repeticiones por umbral\n" % NREP)
    print("  %-8s %7s %14s %20s %10s" % ("umbral", "|T|", "cota", "verdad", "cociente"))
    res = {}
    for thr in THRESHOLDS:
        tgt = objetivo(w, thr)
        cota = float((1.0 / w[tgt]).sum())
        mu, sem = verdad(w, tgt, rng)
        r = cota / mu
        print("  %-8.3f %7d %14.1f %12.1f +- %-5.1f %9.2fx" % (thr, len(tgt), cota, mu, sem, r))
        res["%.3f" % thr] = {"n_objetivo": int(len(tgt)), "cota": round(cota, 2),
                             "verdad": round(float(mu), 2), "sem": round(float(sem), 2),
                             "cociente": round(float(r), 2)}
    print()
    print("  EL PAPER AFIRMA: ~2.7x al 90%% y ~17x al 99.9%%")
    print("  medido:          %.2fx y %.2fx" % (res["0.900"]["cociente"], res["0.999"]["cociente"]))

    salida = {"_nota": ("Cuanto sobreestima la suma del coleccionista al tiempo real de "
                        "coleccion. La seccion 3.1 citaba ~2.7x y ~17x sin generador."),
              "fuente_pesos": "results/coupon_w.dat",
              "repeticiones": NREP, "semilla": SEMILLA,
              "el_paper_imprime": {"0.900": 2.7, "0.999": 17},
              "medido": res}
    io.open(DEST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(salida, indent=1, ensure_ascii=False) + "\n")
    print("\n  escrito %s" % os.path.normpath(DEST))


if __name__ == "__main__":
    main()
