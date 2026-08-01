"""
GFlowNet-augmented SQD recovery under BACKEND-INFORMED noise (Lab2 taxonomy)
===========================================================================
Upgrade over gflownet_qsample.py: replace the naive symmetric bit-flip with the
real noise taxonomy from QGSS Lab 2, parameterized by rates pulled from a Heron-
class FakeBackend when available:
  - depolarizing background  (prob λ a shot is fully randomized),
  - ASYMMETRIC readout error (p10 = 1->0 dominates due to T1, p01 = 0->1),
  - (thermal relaxation is captured by the readout asymmetry toward 0).
Asymmetric noise systematically LOSES electrons -> wrong particle number -> makes
configuration recovery genuinely hard, which is where a constraint-respecting
generator (GFlowNet always emits exactly-N determinants) can earn its keep.

GFlowNet reward is FUSED (recovered empirical freq + cheap EN-PT1) and TEMPERED
(β<1) to prevent the mode collapse seen previously.
"""
import time
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)

# ---------- backend-informed noise rates ----------
LAMBDA = 0.05                    # depolarizing background (fallback)
P10, P01 = 0.020, 0.008          # readout: 1->0 (T1) dominates (fallback, Heron-typical)
src = "fallback Heron-typical"
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    bk = FakeTorino(); props = bk.properties()
    p10s, p01s, g2 = [], [], []
    for q in range(bk.num_qubits):
        try:
            p10s.append(props.qubit_property(q, "prob_meas0_prep1")[0])  # measured 0 | prep 1 = 1->0
            p01s.append(props.qubit_property(q, "prob_meas1_prep0")[0])
        except Exception:
            pass
    if p10s:
        P10 = float(np.median(p10s)); P01 = float(np.median(p01s)); src = "FakeTorino (Heron r1)"
except Exception as e:
    src = f"fallback ({type(e).__name__})"
log(f"noise rates [{src}]: readout 1->0={P10:.4f}  0->1={P01:.4f}  depol λ={LAMBDA}")

# ---------- system ----------
R = 2.5; NCAS, NELECAS = 12, (5, 5); na, nb = NELECAS
mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
str_to_idx = {int(s): i for i, s in enumerate(strs_a)}
set_to_idx = {frozenset(p for p in range(NCAS) if (int(s) >> p) & 1): i for i, s in enumerate(strs_a)}
HF = (1 << na) - 1; hf_idx = str_to_idx[HF]
exc = np.array([na - bin(int(s) & HF).count("1") for s in strs_a])

civ_hf = np.zeros((dim_a, dim_a)); civ_hf[hf_idx, hf_idx] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ_hf, NCAS, NELECAS).reshape(dim_a, dim_a)
hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)
den = Hc[hf_idx, hf_idx] - hdiag; den[hf_idx, hf_idx] = 1.0
c1 = Hc / den; c1[hf_idx, hf_idx] = 1.0
w_cheap = (c1**2).sum(1); w_cheap = w_cheap / w_cheap.sum()

e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_a); w_true = (civec**2).sum(1); w_true /= w_true.sum()
order_true = np.argsort(w_true)[::-1]; K = 60; top_true = set(order_true[:K].tolist())
log(f"R={R}A FCI={e_fci:.6f} triples+={100*w_true[exc>=3].sum():.1f}%")

_sci = selected_ci.SelectedCI()
def E(strings):
    s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return float(out[0] if isinstance(out, (tuple, list)) else out)

# ---------- backend-informed noisy sampler ----------
def quantum_shots(n):
    ideal = rng.choice(dim_a, size=n, p=w_true)
    bits = np.array([[(int(strs_a[i]) >> p) & 1 for p in range(NCAS)] for i in ideal], dtype=np.int8)
    depol = rng.random(n) < LAMBDA                       # fully-randomized shots
    bits[depol] = (rng.random((depol.sum(), NCAS)) < 0.5).astype(np.int8)
    ones, zeros = bits == 1, bits == 0                   # asymmetric readout
    bits[ones & (rng.random(bits.shape) < P10)] = 0
    bits[zeros & (rng.random(bits.shape) < P01)] = 1
    return bits
def b2s(b): return int(sum(int(v) << p for p, v in enumerate(b)))
def recover_ibm(bits, occ):
    out = []
    for row in bits:
        s = row.copy(); m = int(s.sum())
        if m == na: out.append(b2s(s)); continue
        if m > na:
            occd = np.where(s == 1)[0]; s[occd[np.argsort(occ[occd])[:m-na]]] = 0
        else:
            emp = np.where(s == 0)[0]; s[emp[np.argsort(occ[emp])[::-1][:na-m]]] = 1
        out.append(b2s(s))
    return out

# ---------- GFlowNet (tempered fused reward) ----------
class Policy(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n))
        s.logZ = nn.Parameter(torch.zeros(1))
    def forward(s, x): return s.net(x)
def sample_batch(net, B):
    state = torch.zeros(B, NCAS); logPF = torch.zeros(B)
    for _ in range(na):
        lp = torch.log_softmax(net(state).masked_fill(state.bool(), -1e9), 1)
        a = torch.distributions.Categorical(logits=lp).sample()
        logPF += lp.gather(1, a[:, None]).squeeze(1)
        state = state.scatter(1, a[:, None], 1.0)
    idxs = np.array([set_to_idx[frozenset(np.where(state[b].numpy() > 0)[0].tolist())] for b in range(B)])
    return idxs, logPF
def train_gflownet(reward, iters=1500):
    net = Policy(NCAS)
    opt = torch.optim.Adam([{"params": net.net.parameters(), "lr": 1e-3}, {"params": [net.logZ], "lr": 1e-1}])
    Rt = torch.tensor(reward / reward.sum(), dtype=torch.float32)
    for it in range(iters):
        idxs, logPF = sample_batch(net, 256)
        loss = ((net.logZ + logPF - torch.log(Rt[idxs] + 1e-12)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net
@torch.no_grad()
def gfn_draws(net, n):
    o = []
    while len(o) < n:
        idxs, _ = sample_batch(net, 512); o += idxs.tolist()
    return np.array(o[:n])

# ---------- experiment ----------
D = 120; BETA_T = 0.5
def topD(counts):
    idxs = [i for i, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:D]
    ntrip = int((exc[idxs] >= 3).sum()); ntop = len(set(idxs) & top_true)
    return (E([strs_a[i] for i in idxs]) - e_fci) * 1000, len(idxs), ntrip, ntop
cheap_sel = {i: w_cheap[i] for i in np.argsort(w_cheap)[::-1][:D]}
log(f"\n{'shots':>6s}  {'method':10s} {'Eerr(mHa)':>10s} {'dim':>4s} {'trip+':>5s} {'top60':>6s} {'lost%':>6s}")
for shots in (200, 500, 1000, 2000):
    bits = quantum_shots(shots)
    good = np.array([int(b.sum()) == na for b in bits])
    counts_raw = {}
    for b in bits[good]:
        i = str_to_idx[b2s(b)]; counts_raw[i] = counts_raw.get(i, 0) + 1
    occ = bits[good].mean(0) if good.any() else np.full(NCAS, na/NCAS)
    counts_ibm = {}
    for s in recover_ibm(bits, occ):
        i = str_to_idx[s]; counts_ibm[i] = counts_ibm.get(i, 0) + 1
    f_emp = np.zeros(dim_a)
    for i, c in counts_ibm.items(): f_emp[i] = c
    f_emp = f_emp / max(f_emp.sum(), 1)
    reward = np.maximum(f_emp + 0.1 * w_cheap / w_cheap.max() * max(f_emp.max(), 1e-9), 1e-9) ** BETA_T
    net = train_gflownet(reward)
    gcounts = {}
    for i in gfn_draws(net, 20000): gcounts[int(i)] = gcounts.get(int(i), 0) + 1
    lost = 100 * (1 - good.mean())
    for name, counts in [("raw", counts_raw), ("ibm", counts_ibm), ("cheap", cheap_sel), ("gflownet", gcounts)]:
        e, d, tp, tt = topD(counts)
        extra = f"{lost:6.1f}" if name == "raw" else ""
        log(f"{shots:6d}  {name:10s} {e:10.2f} {d:4d} {tp:5d} {tt:4d}/{K} {extra:>6s}")
    log("")
log("Verdict: with realistic asymmetric noise, does gflownet (always valid-N, tempered)")
log("close the gap vs raw/ibm — especially where 'lost%' (discarded corrupted shots) is high?")
log("DONE.")
