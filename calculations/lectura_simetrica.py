"""Symmetric-readout control for the noise result: five seeds, and a deposited output.

WHAT THE MANUSCRIPT CLAIMS NOW, and what it used to. Section 7 prints the output of
this script: the generative proposer degrading to 47.9 +- 3.2 mHa against 31.1 +- 3.9 for
the classical recovery loop and 33.5 +- 2.8 raw, five seeds, t_4 = 10.2. Before this
script existed it printed 40.9 / 32.4 / 33.1, and those are retracted. It is the single
caveat that most limits the paper's one positive result, so it is the one that most
needed to be regenerable.

TWO THINGS WERE WRONG WITH HOW IT STOOD.

  1. It had no deposited output. calculations/gflownet_qsample.py runs this experiment
     but writes nothing, so the three numbers could not be checked against anything.
  2. It was a SINGLE SEED, in a paper whose own benchmarking element (v) says a single
     run of a seed-dependent proposer is an anecdote. The GFlowNet here is exactly that.

This script fixes both: five seeds at the quoted setting, mean +- sample s.d., written
to results/lectura_simetrica.json.

THE SETTING. N2 at R = 2.5 A, CAS(10e,12o)/cc-pVDZ, 2000 shots, subspace dimension
D = 120. The readout is a SYMMETRIC bit flip at p = 0.03 per spin-orbital -- that is
the whole point of the control: the backend-calibrated model of section 7 is asymmetric
(1->0 dominates through T1), which systematically LOSES electrons, and the generative
proposer's advantage there comes from emitting exactly-N determinants by construction.
Make the channel symmetric and that structural advantage largely disappears.

GAUGE. mol.symmetry=True, unlike the original script. Canonical RHF orbitals are not
reproducible inside the degenerate pi shells (section 2.1), and every string count and
subspace below moves with the orientation. Run with OMP_NUM_THREADS=1.

    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-nb \
        python /w/calculations/lectura_simetrica.py
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

R = 2.5
NCAS, NELECAS = 12, (5, 5)
na, nb = NELECAS
P_FLIP = 0.03          # SYMMETRIC per-spin-orbital bit flip: the control
SHOTS = 2000
D = 120
SEEDS = [0, 1, 2, 3, 4]
ITERS = 1500
NDRAW = 20000
K = 60

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "results", "lectura_simetrica.json")

# ---------------------------------------------------------------- the system
mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas()
h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
strs_a = cistring.make_strings(range(NCAS), na)
dim_a = len(strs_a)
str_to_idx = {int(s): i for i, s in enumerate(strs_a)}
set_to_idx = {frozenset(p for p in range(NCAS) if (int(s) >> p) & 1): i
              for i, s in enumerate(strs_a)}
HF = (1 << na) - 1
hf_idx = str_to_idx[HF]
exc = np.array([na - bin(int(s) & HF).count("1") for s in strs_a])

civ_hf = np.zeros((dim_a, dim_a))
civ_hf[hf_idx, hf_idx] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ_hf, NCAS, NELECAS).reshape(dim_a, dim_a)
hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)
den = Hc[hf_idx, hf_idx] - hdiag
den[hf_idx, hf_idx] = 1.0
c1 = Hc / den
c1[hf_idx, hf_idx] = 1.0
w_cheap = (c1 ** 2).sum(1)
w_cheap /= w_cheap.sum()

e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_a)
w_true = (civec ** 2).sum(1)
w_true /= w_true.sum()
top_true = set(np.argsort(w_true)[::-1][:K].tolist())
log(f"N2 R={R} A  E_FCI={e_fci:.9f} Ha  gauge={mol.groupname}  "
    f"triples+ true weight={100 * w_true[exc >= 3].sum():.1f}%")

_sci = selected_ci.SelectedCI()


def E(strings):
    s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return float(out[0] if isinstance(out, (tuple, list)) else out)


def topD_energy(counts):
    idxs = [i for i, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:D]
    e = (E([strs_a[i] for i in idxs]) - e_fci) * 1000
    return e, len(idxs), int((exc[idxs] >= 3).sum()), len(set(idxs) & top_true)


def bits_to_str(b):
    return int(sum(int(v) << p for p, v in enumerate(b)))


def quantum_shots(n, rng):
    """n ideal strings drawn from |c|^2, then a SYMMETRIC bit flip on every spin-orbital."""
    ideal = rng.choice(dim_a, size=n, p=w_true)
    bits = np.array([[(int(strs_a[i]) >> p) & 1 for p in range(NCAS)] for i in ideal])
    return bits ^ (rng.random((n, NCAS)) < P_FLIP)


def recover_ibm(bits, occ):
    """Occupancy-based recovery: restore every row to exactly na set bits."""
    out = []
    for row in bits:
        s = row.copy()
        m = int(s.sum())
        if m == na:
            out.append(bits_to_str(s))
            continue
        if m > na:
            occd = np.where(s == 1)[0]
            s[occd[np.argsort(occ[occd])[:m - na]]] = 0
        else:
            emp = np.where(s == 0)[0]
            s[emp[np.argsort(occ[emp])[::-1][:na - m]]] = 1
        out.append(bits_to_str(s))
    return out


class Policy(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 256),
                              nn.ReLU(), nn.Linear(256, n))
        s.logZ = nn.Parameter(torch.zeros(1))

    def forward(s, x):
        return s.net(x)


def sample_batch(net, B):
    state = torch.zeros(B, NCAS)
    logPF = torch.zeros(B)
    for _ in range(na):
        lp = torch.log_softmax(net(state).masked_fill(state.bool(), -1e9), 1)
        a = torch.distributions.Categorical(logits=lp).sample()
        logPF += lp.gather(1, a[:, None]).squeeze(1)
        state = state.scatter(1, a[:, None], 1.0)
    idxs = np.array([set_to_idx[frozenset(np.where(state[b].numpy() > 0)[0].tolist())]
                     for b in range(B)])
    return idxs, logPF


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    bits = quantum_shots(SHOTS, rng)
    valid = np.array([int(b.sum()) == na for b in bits])
    counts_raw = {}
    for b in bits[valid]:
        i = str_to_idx[bits_to_str(b)]
        counts_raw[i] = counts_raw.get(i, 0) + 1
    occ = bits[valid].mean(0) if valid.any() else np.full(NCAS, na / NCAS)

    counts_ibm = {}
    for s in recover_ibm(bits, occ):
        i = str_to_idx[s]
        counts_ibm[i] = counts_ibm.get(i, 0) + 1

    f_emp = np.zeros(dim_a)
    for i, c in counts_ibm.items():
        f_emp[i] = c
    f_emp /= max(f_emp.sum(), 1)
    reward = np.maximum(f_emp + 0.1 * w_cheap / w_cheap.max() * f_emp.max(), 1e-9)

    net = Policy(NCAS)
    opt = torch.optim.Adam([{"params": net.net.parameters(), "lr": 1e-3},
                            {"params": [net.logZ], "lr": 1e-1}])
    Rt = torch.tensor(reward / reward.sum(), dtype=torch.float32)
    for _ in range(ITERS):
        idxs, logPF = sample_batch(net, 256)
        loss = ((net.logZ + logPF - torch.log(Rt[idxs] + 1e-12)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    gcounts = {}
    with torch.no_grad():
        drawn = []
        while len(drawn) < NDRAW:
            idxs, _ = sample_batch(net, 512)
            drawn += idxs.tolist()
    for i in drawn[:NDRAW]:
        gcounts[int(i)] = gcounts.get(int(i), 0) + 1

    out = {"lost_pct": 100.0 * (1 - valid.mean())}
    for name, counts in (("raw", counts_raw), ("ibm", counts_ibm), ("gflownet", gcounts)):
        e, d, tp, tt = topD_energy(counts)
        out[name] = {"err_mHa": e, "dim": d, "triples_plus": tp, "top60": tt}
    return out


def main():
    per_seed = []
    for s in SEEDS:
        r = run_seed(s)
        per_seed.append(r)
        log("seed %d: lost %.1f%%  raw %.2f  ibm %.2f  gflownet %.2f  (mHa)"
            % (s, r["lost_pct"], r["raw"]["err_mHa"], r["ibm"]["err_mHa"],
               r["gflownet"]["err_mHa"]))

    res = {"system": "N2", "R_ang": R, "active": "(10e,12o)", "basis": "cc-pvdz",
           "gauge": "%s symmetry-adapted, OMP_NUM_THREADS=1" % mol.groupname,
           "readout": "SYMMETRIC bit flip, p=%.3f per spin-orbital" % P_FLIP,
           "shots": SHOTS, "D": D, "seeds": SEEDS, "gflownet_iters": ITERS,
           "E_FCI_Ha": e_fci, "per_seed": per_seed, "summary": {}}

    print()
    print("=" * 70)
    print("  %-10s %-18s %-10s" % ("metodo", "error (mHa)", "t vs gflownet"))
    g = np.array([p["gflownet"]["err_mHa"] for p in per_seed])
    for name in ("raw", "ibm", "gflownet"):
        v = np.array([p[name]["err_mHa"] for p in per_seed])
        res["summary"][name] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                "n": len(v)}
        if name == "gflownet":
            print("  %-10s %6.2f +- %-8.2f %s" % (name, v.mean(), v.std(ddof=1), "--"))
        else:
            d = g - v
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else float("nan")
            print("  %-10s %6.2f +- %-8.2f t4 = %+.1f" % (name, v.mean(), v.std(ddof=1), t))
            res["summary"][name]["t_vs_gflownet"] = float(t)
    print("=" * 70)
    print("  El manuscrito cita 40.9 (generativo), 32.4 (ibm) y 33.1 (crudo), de UNA")
    print("  semilla y sin fijar el gauge. Las cifras de arriba son las que hay que")
    print("  imprimir: cinco semillas, gauge fijado, y con su dispersion.")
    print("  ORDENAMIENTO: el clasico %s al generativo."
          % ("GANA" if res["summary"]["ibm"]["mean"] < res["summary"]["gflownet"]["mean"]
             else "PIERDE frente"))

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("\n  escrito %s" % os.path.normpath(OUT))


if __name__ == "__main__":
    main()
