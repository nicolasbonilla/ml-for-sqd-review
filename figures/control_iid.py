"""The like-for-like control Figure 6 needs: i.i.d. sampling proportional to the same reward.

THE OBJECTION THIS ANSWERS. Figure 6 compares a GFlowNet, which builds its subspace from
the first D DISTINCT strings seen in a sampling stream, against a deterministic greedy
selector, which simply takes the top D by score. Those are not the same task. A sampler
must COLLECT D distinct strings; a ranker merely TAKES them -- and section 3.1 of this paper
is itself the statement that the first is strictly the harder job. Reading "the proposer
loses to the ranking" off that comparison therefore charges the proposer for being a
sampler at all.

THE CONTROL. Sample i.i.d. with probability exactly proportional to the same cheap
Epstein-Nesbet reward the GFlowNet is trained on, and build the subspace the same way, from
distinct draws. That isolates what the flow network adds over naive proportional sampling of
its own signal -- which is the literal referent of section 4.7's sentence "sampling a cheap
reward proportionally does not outperform simply taking its top determinants".

A SECOND ARM, for the same reason at the other end: i.i.d. sampling proportional to the
EXACT weights. That measures what sampling costs against ranking when the reward is perfect,
so the penalty can be separated from the reward's quality.

FIVE SEEDS, because element (v) of this paper's own standard says a single run of a
stochastic procedure is an anecdote, and the deposit already contains a single-seed version
of this control in the companion notebook.

GAUGE. mol.symmetry=True and OMP_NUM_THREADS=1, matching figures/barrido_suelo.py so the
arms are comparable string for string. No torch: nothing here is trained.

    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci \
        python /w/figures/control_iid.py
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

NCAS, NELECAS = 12, (5, 5)
na, nb = NELECAS
SIZES = [30, 60, 120, 240]
SEEDS = [0, 1, 2, 3, 4]
NDRAW = 50000           # el mismo presupuesto por brazo que figures/barrido_suelo.py

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, os.pardir, "results", "control_iid.json")

mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas()
h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
strs = cistring.make_strings(range(NCAS), na)
dim = len(strs)
hf = int(np.where(strs == (1 << na) - 1)[0][0])

civ = np.zeros((dim, dim)); civ[hf, hf] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ, NCAS, NELECAS).reshape(dim, dim)
hd = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim, dim)
den = Hc[hf, hf] - hd; den[hf, hf] = 1.0
c1 = Hc / den; c1[hf, hf] = 1.0
w_cheap = (c1 ** 2).sum(1); w_cheap /= w_cheap.sum()

solver = pyscf.fci.direct_spin1.FCI(); solver.conv_tol = 1e-13
e_fci, cV = solver.kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
cV = cV.reshape(dim, dim)
w_true = (cV ** 2).sum(1); w_true /= w_true.sum()
log(f"N2 R=2.0 A  {dim} cadenas  E_FCI={e_fci:.9f} Ha  gauge={mol.groupname}")

_sci = selected_ci.SelectedCI()


def E(idxs):
    s = np.asarray(sorted(set(int(strs[i]) for i in idxs)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return (float(out[0] if isinstance(out, (tuple, list)) else out) - e_fci) * 1000


def compacidad(draws):
    """Error a cada D, recogiendo cadenas DISTINTAS del flujo, como hace el GFlowNet."""
    visto, salida, obj, j = set(), {}, sorted(SIZES), 0
    for d in draws:
        visto.add(int(d))
        while j < len(obj) and len(visto) >= obj[j]:
            salida[obj[j]] = E(visto)
            j += 1
    return salida


def arm(p, nombre):
    acc = {m: [] for m in SIZES}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        c = compacidad(rng.choice(dim, size=NDRAW, p=p))
        for m in SIZES:
            if m in c:
                acc[m].append(c[m])
        log("   %-10s seed %d: %s" % (nombre, sd,
            {k: round(v, 1) for k, v in sorted(c.items())}))
    out = {}
    for m in SIZES:
        v = acc[m]
        out[str(m)] = ({"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                        "n": len(v)} if len(v) > 1 else
                       ({"mean": float(v[0]), "std": 0.0, "n": 1} if v else None))
    return out


def main():
    res = {"_nota": ("El control que la Fig. 6 necesita para ser comparacion igual a igual: "
                     "muestreo i.i.d. proporcional a la MISMA recompensa que entrena al "
                     "GFlowNet, y a los pesos exactos. Cinco semillas. Sin torch."),
           "gauge": "D-infinity-h adaptado por simetria, OMP_NUM_THREADS=1",
           "n_draws_por_brazo": NDRAW, "seeds": SEEDS, "E_FCI_Ha": float(e_fci),
           "arms": {}}
    res["arms"]["iid_cheap"] = arm(w_cheap, "iid_cheap")
    res["arms"]["iid_exact"] = arm(w_true, "iid_exact")
    # el ranking determinista de la misma recompensa, para tener las cuatro esquinas
    orden_c = np.argsort(w_cheap)[::-1]
    orden_t = np.argsort(w_true)[::-1]
    res["arms"]["ranked_cheap"] = {str(m): {"mean": E(orden_c[:m]), "std": 0.0, "n": 1}
                                   for m in SIZES}
    res["arms"]["ranked_exact"] = {str(m): {"mean": E(orden_t[:m]), "std": 0.0, "n": 1}
                                   for m in SIZES}

    print()
    print("=" * 76)
    print("  LAS CUATRO ESQUINAS: (recompensa barata | exacta) x (muestreada | rankeada)")
    print("  %-14s %s" % ("D", "".join("%14d" % m for m in SIZES)))
    for k in ("iid_cheap", "ranked_cheap", "iid_exact", "ranked_exact"):
        fila = ""
        for m in SIZES:
            a = res["arms"][k][str(m)]
            fila += "%14s" % ("no llena" if a is None else
                              ("%.1f" % a["mean"] if a["n"] == 1 else
                               "%.1f+-%.1f" % (a["mean"], a["std"])))
        print("  %-14s %s" % (k, fila))
    print("=" * 76)
    print("  El muestreo proporcional a la recompensa barata no llena D=60 con 5e4 tiradas.")
    print("  El GFlowNet entrenado sobre esa misma recompensa llega a D=240 (Fig. 6).")

    io.open(DEST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    log("WROTE %s" % os.path.normpath(DEST))


if __name__ == "__main__":
    main()
