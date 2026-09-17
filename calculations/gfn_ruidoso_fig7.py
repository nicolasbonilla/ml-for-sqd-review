"""The noisy generative bars of Figure 7, which until now had no generator at all.

WHAT FIGURE 7 SHOWS. Four bars: classical selected CI (heat-bath) against the
GFlowNet-SQD proposer under simulated Heron noise, on H2O and on N2, all at subspace
dimension D = 120 where the exact FCI answer is computable. It is the paper's "decisive
test at FCI-verifiable scale", and section 5.5 rests its verdict on it.

THE DEFECT THIS FIXES. The two classical bars come from calculations/hci_baseline.py.
The two GENERATIVE bars -- 2.1 mHa on H2O and 26.7 on N2 -- were hand-typed literals.
They appear as `gfn=2.1` and `gfn=26.7` in hci_baseline.py's molecule table, as a
hardcoded list in figures/hci_fig.py, and as coordinates in the TikZ of main.tex. Nothing
in the deposit computed them. For a figure whose whole purpose is to make a comparison
falsifiable, half of it could not be checked.

WHAT THIS SCRIPT DOES. Runs the same pipeline as the noise ladder of section 7, at the
geometries of Figure 7, and deposits the result:

  1. Draw SHOTS samples from the exact |c_alpha|^2 (a noiseless quantum sampler).
  2. Corrupt them with the Heron-calibrated model: a depolarizing background that fully
     randomises a shot with probability LAMBDA, then ASYMMETRIC readout, 1->0 at P10
     (T1-dominated) and 0->1 at P01. Rates are pulled from FakeTorino when qiskit is
     present and fall back to the published medians otherwise.
  3. Repair every broken shot by occupation-weighted recovery (the S-CORE analogue).
  4. Train a GFlowNet on the fused, tempered reward -- recovered empirical frequency plus
     the cheap Epstein-Nesbet prior -- and take its top-D distinct strings.
  5. Diagonalize in that subspace and report the error against exact FCI.

FIVE SEEDS, because the paper's own benchmarking element (v) says a single run of a
seed-dependent proposer is an anecdote, and this script exists precisely because that
element was not applied here.

GAUGE. mol.symmetry=True. The subspace SELECTION moves with the orbital gauge, so the
D = 120 energy does too -- on N2 the classical bar moves by 0.7 mHa between an unpinned
and a pinned run, about half of chemical accuracy. Run single-threaded:

    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-nb \
        python /w/calculations/gfn_ruidoso_fig7.py
"""
import io
import json
import os
import time

import numpy as np
import pyscf
import torch
import torch.nn as nn
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.fci import cistring, direct_spin1, selected_ci

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

SHOTS, D, BETA_T, ITERS, NDRAW = 1000, 120, 0.5, 800, 20000
SEEDS = [0, 1, 2, 3, 4]

LAMBDA0, P100, P010 = 0.05, 0.0229, 0.0200
FUENTE = "published FakeTorino medians"
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    _pr, _nq = FakeTorino().properties(), FakeTorino().num_qubits
    P100 = float(np.median([_pr.qubit_property(q, "prob_meas0_prep1")[0] for q in range(_nq)]))
    P010 = float(np.median([_pr.qubit_property(q, "prob_meas1_prep0")[0] for q in range(_nq)]))
    FUENTE = "FakeTorino (Heron r1), live"
except Exception:
    pass

# Las mismas geometrias y espacios activos que calculations/hci_baseline.py, para que las
# barras clasicas y las generativas de la Fig. 7 sean comparables barra a barra.
MOLS = {
    "h2o": dict(atom="O 0 0 0; H 0 0.98 0.76; H 0 -0.98 0.76", ncore=1, ncas=12, nelecas=(4, 4)),
    "n2": dict(atom="N 0 0 0; N 0 0 2.0", ncore=2, ncas=12, nelecas=(5, 5)),
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, os.pardir, "results", "gfn_ruidoso_fig7.json")


class Policy(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 256),
                              nn.ReLU(), nn.Linear(256, n))
        s.logZ = nn.Parameter(torch.zeros(1))

    def forward(s, x):
        return s.net(x)


def corre(nombre, S):
    NCAS, NELECAS = S["ncas"], S["nelecas"]
    na = NELECAS[0]
    mol = gto.M(atom=S["atom"], basis="cc-pvdz", symmetry=True, verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
    cas = mcscf.CASCI(mf, NCAS, NELECAS); cas.ncore = S["ncore"]
    h1, ecore = cas.get_h1cas()
    h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)

    strs = cistring.make_strings(range(NCAS), na)
    dim = len(strs)
    s2i = {int(x): i for i, x in enumerate(strs)}
    set2i = {frozenset(p for p in range(NCAS) if (int(x) >> p) & 1): i
             for i, x in enumerate(strs)}
    hf = s2i[(1 << na) - 1]

    v0 = np.zeros((dim, dim)); v0[hf, hf] = 1.0
    h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
    Hc = direct_spin1.contract_2e(h2e, v0, NCAS, NELECAS).reshape(dim, dim)
    hd = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim, dim)
    den = Hc[hf, hf] - hd; den[hf, hf] = 1.0
    cc = Hc / den; cc[hf, hf] = 1.0
    w_cheap = (cc ** 2).sum(1); w_cheap /= w_cheap.sum()
    cheap_order = np.argsort(w_cheap)[::-1]

    solver = pyscf.fci.direct_spin1.FCI(); solver.conv_tol = 1e-13
    e_fci, cv = solver.kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
    cv = cv.reshape(dim, dim)
    w_true = (cv ** 2).sum(1); w_true /= w_true.sum()
    log(f"{nombre}: CAS({sum(NELECAS)}e,{NCAS}o)  {dim} cadenas  E_FCI={e_fci:.9f} Ha  "
        f"gauge={mol.groupname}")

    _sci = selected_ci.SelectedCI()

    def E(idxs):
        s = np.asarray(sorted(set(int(strs[i]) for i in idxs)), dtype=np.int64)
        out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
        return (float(out[0] if isinstance(out, (tuple, list)) else out) - e_fci) * 1000

    def b2s(b):
        return int(sum(int(v) << p for p, v in enumerate(b)))

    def una(seed):
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        ideal = rng.choice(dim, size=SHOTS, p=w_true)
        bits = np.array([[(int(strs[i]) >> p) & 1 for p in range(NCAS)] for i in ideal],
                        dtype=np.int8)
        dep = rng.random(SHOTS) < LAMBDA0
        bits[dep] = (rng.random((int(dep.sum()), NCAS)) < 0.5).astype(np.int8)
        o, z = bits == 1, bits == 0
        bits[o & (rng.random(bits.shape) < P100)] = 0
        bits[z & (rng.random(bits.shape) < P010)] = 1

        good = np.array([int(b.sum()) == na for b in bits])
        occ = bits[good].mean(0) if good.any() else np.full(NCAS, na / NCAS)
        rec = []
        for row in bits:
            s = row.copy(); m = int(s.sum())
            if m == na:
                rec.append(b2s(s)); continue
            if m > na:
                od = np.where(s == 1)[0]; s[od[np.argsort(occ[od])[:m - na]]] = 0
            else:
                em = np.where(s == 0)[0]; s[em[np.argsort(occ[em])[::-1][:na - m]]] = 1
            rec.append(b2s(s))
        cibm = {}
        for s in rec:
            i = s2i[s]; cibm[i] = cibm.get(i, 0) + 1

        f = np.zeros(dim)
        for i, n in cibm.items():
            f[i] = n
        f /= max(f.sum(), 1)
        rew = np.maximum(f + 0.1 * w_cheap / w_cheap.max() * max(f.max(), 1e-9), 1e-9) ** BETA_T

        net = Policy(NCAS)
        opt = torch.optim.Adam([{"params": net.net.parameters(), "lr": 1e-3},
                                {"params": [net.logZ], "lr": 1e-1}])
        Rt = torch.tensor(rew / rew.sum(), dtype=torch.float32)

        def lote(Bn):
            st = torch.zeros(Bn, NCAS); lpf = torch.zeros(Bn)
            for _ in range(na):
                lp = torch.log_softmax(net(st).masked_fill(st.bool(), -1e9), 1)
                a = torch.distributions.Categorical(logits=lp).sample()
                lpf += lp.gather(1, a[:, None]).squeeze(1)
                st = st.scatter(1, a[:, None], 1.0)
            return np.array([set2i[frozenset(np.where(st[b].numpy() > 0)[0].tolist())]
                             for b in range(Bn)]), lpf

        for _ in range(ITERS):
            idx, lpf = lote(256)
            loss = ((net.logZ + lpf - torch.log(Rt[idx] + 1e-12)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        g = {}
        with torch.no_grad():
            got = []
            while len(got) < NDRAW:
                idx, _ = lote(512); got += idx.tolist()
        for i in got[:NDRAW]:
            g[int(i)] = g.get(int(i), 0) + 1

        rank = lambda c: [i for i, _ in sorted(c.items(), key=lambda kv: -kv[1])]
        # el control clasico justo: recuperacion IBM, y se rellena con el prior barato
        sel = list(dict.fromkeys(rank(cibm)))[:D]
        for i in cheap_order:
            if len(sel) >= D:
                break
            if i not in sel:
                sel.append(int(i))
        return E(sel[:D]), E(rank(g)[:D]), 100.0 * (1 - good.mean())

    clas, gen, perd = [], [], []
    for sd in SEEDS:
        a, b, l = una(sd)
        clas.append(a); gen.append(b); perd.append(l)
        log(f"   seed {sd}: ibm+cheap {a:6.2f}   gflownet {b:6.2f} mHa   ({l:.0f}% lost)")
    c, g = np.array(clas), np.array(gen)
    # Los dos brazos salen de la MISMA llamada a una(sd) y por tanto de los mismos tiros
    # ruidosos: el emparejamiento por semilla es legitimo por construccion, y sin los pares
    # depositados el estadistico emparejado que el paper cita no es adjudicable por nadie.
    dif = c - g                      # positivo = gana el proponente generativo
    sd_dif = float(dif.std(ddof=1))
    n = len(dif)
    t_par = float(dif.mean() * np.sqrt(n) / sd_dif) if sd_dif > 0 else float("inf")
    return {"atom": S["atom"], "ncas": NCAS, "nelecas": list(NELECAS), "D": D,
            "shots": SHOTS, "seeds": SEEDS, "E_FCI_Ha": float(e_fci),
            "lost_pct": float(np.mean(perd)),
            "gflownet": {"mean": float(g.mean()), "std": float(g.std(ddof=1)),
                         "per_seed": [float(x) for x in g]},
            "ibm_cheap": {"mean": float(c.mean()), "std": float(c.std(ddof=1)),
                          "per_seed": [float(x) for x in c]},
            "pareado": {"_nota": ("diferencia clasico - generativo, semilla a semilla; "
                                  "positiva = gana el proponente generativo"),
                        "por_semilla": [float(x) for x in dif],
                        "media_mHa": float(dif.mean()), "sd_mHa": sd_dif,
                        "t": t_par, "n": n,
                        "todas_del_mismo_signo": bool(np.all(dif > 0) or np.all(dif < 0))}}


def main():
    log(f"noise: readout 1->0 = {P100:.4f}, 0->1 = {P010:.4f}, depol lambda = {LAMBDA0}"
        f"   [{FUENTE}]")
    res = {"_nota": ("Barras generativas de la Fig. 7. Hasta el 2026-09-14 los valores "
                     "2.1 y 26.7 mHa estaban escritos a mano y ningun script del deposito "
                     "los producia."),
           "noise": {"P_1to0": P100, "P_0to1": P010, "lambda_depol": LAMBDA0,
                     "source": FUENTE},
           "gauge": "D-infinity-h adaptado por simetria, OMP_NUM_THREADS=1", "mols": {}}
    for nombre, S in MOLS.items():
        res["mols"][nombre] = corre(nombre, S)

    print()
    print("=" * 72)
    print("  %-6s %-22s %-22s %s" % ("", "GFlowNet (noisy)", "ibm+cheap control", "tiros perdidos"))
    for k, v in res["mols"].items():
        print("  %-6s %8.2f +- %-11.2f %8.2f +- %-11.2f %.0f%%"
              % (k, v["gflownet"]["mean"], v["gflownet"]["std"],
                 v["ibm_cheap"]["mean"], v["ibm_cheap"]["std"], v["lost_pct"]))
    print("=" * 72)
    print("  El manuscrito imprimia 2.1 (H2O) y 26.7 (N2) sin generador. Estos son los")
    print("  valores medidos, con su dispersion sobre cinco semillas.")

    io.open(DEST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    log("WROTE %s" % os.path.normpath(DEST))


if __name__ == "__main__":
    main()
