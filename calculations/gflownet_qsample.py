"""
GFlowNet-augmented configuration recovery for SQD  (flagship MVE, direction A)
==============================================================================
Setting: an (approximate) quantum state is sampled with a FINITE number of noisy
shots (bit-flip noise -> wrong particle number). SQD must recover a determinant
subspace from these corrupted samples. We compare configuration-recovery methods:

  raw       : keep only shots with correct electron number (discard corrupted).
  ibm       : occupancy-based recovery of corrupted shots (qiskit-addon-sqd style:
              flip toward the average occupation to restore correct N).
  cheap     : deterministic top-K by a cheap Epstein-Nesbet S+D reward (no shots;
              a classical ceiling, structurally blind to triples+).
  gflownet  : a GFlowNet that (i) generates ONLY correct-N determinants by
              construction, and (ii) is trained on a reward FUSING the recovered
              empirical sample frequency (carries triples+ signal) with the cheap
              reward (fills S+D gaps). It generalizes the finite sample.

Ground truth: FCI on N2/CAS(10e,12o), stretched (triples+ matter). Metric:
FCI energy error vs QUANTUM SHOTS (the expensive resource) at fixed subspace dim,
plus how many important triples+ each method recovers.
"""
import time
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)

# ---------- system ----------
R = 2.5
NCAS, NELECAS = 12, (5, 5); na, nb = NELECAS
mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
str_to_idx = {int(s): i for i, s in enumerate(strs_a)}
set_to_idx = {frozenset(p for p in range(NCAS) if (int(s) >> p) & 1): i
              for i, s in enumerate(strs_a)}
HF = (1 << na) - 1; hf_idx = str_to_idx[HF]
exc = np.array([na - bin(int(s) & HF).count("1") for s in strs_a])

# cheap Epstein-Nesbet PT1 reward
civ_hf = np.zeros((dim_a, dim_a)); civ_hf[hf_idx, hf_idx] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ_hf, NCAS, NELECAS).reshape(dim_a, dim_a)
hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)
den = Hc[hf_idx, hf_idx] - hdiag; den[hf_idx, hf_idx] = 1.0
c1 = Hc / den; c1[hf_idx, hf_idx] = 1.0
w_cheap = (c1**2).sum(1); w_cheap = w_cheap / w_cheap.sum()

# FCI ground truth (scoring + defines the ideal quantum distribution)
e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_a); w_true = (civec**2).sum(1); w_true /= w_true.sum()
order_true = np.argsort(w_true)[::-1]; K = 60; top_true = set(order_true[:K].tolist())
log(f"R={R}A  FCI={e_fci:.6f}  triples+ true weight={100*w_true[exc>=3].sum():.1f}%")

_sci = selected_ci.SelectedCI()
def E(strings):
    s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return float(out[0] if isinstance(out, (tuple, list)) else out)

# ---------- noisy quantum sampler + recovery ----------
P_FLIP = 0.03
def quantum_shots(n):
    """Draw n ideal strings ∝ w_true, then bit-flip each of NCAS bits w.p. P_FLIP."""
    ideal = rng.choice(dim_a, size=n, p=w_true)
    bits = np.array([[(int(strs_a[i]) >> p) & 1 for p in range(NCAS)] for i in ideal])
    flip = rng.random((n, NCAS)) < P_FLIP
    bits = bits ^ flip
    return bits                                    # (n, NCAS) possibly wrong count

def bits_to_str(b): return int(sum(int(v) << p for p, v in enumerate(b)))

def recover_ibm(bits, occ):
    """Occupancy-based recovery: restore each row to exactly na set bits."""
    out = []
    for row in bits:
        s = row.copy(); m = int(s.sum())
        if m == na: out.append(bits_to_str(s)); continue
        if m > na:                                  # remove lowest-occupancy occupied
            occd = np.where(s == 1)[0]
            drop = occd[np.argsort(occ[occd])[:m - na]]; s[drop] = 0
        else:                                       # add highest-occupancy empty
            emp = np.where(s == 0)[0]
            add = emp[np.argsort(occ[emp])[::-1][:na - m]]; s[add] = 1
        out.append(bits_to_str(s))
    return out

# ---------- GFlowNet ----------
class Policy(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 256),
                              nn.ReLU(), nn.Linear(256, n))
        s.logZ = nn.Parameter(torch.zeros(1))
    def forward(s, x): return s.net(x)
def sample_batch(net, B):
    state = torch.zeros(B, NCAS); logPF = torch.zeros(B)
    for _ in range(na):
        lp = torch.log_softmax(net(state).masked_fill(state.bool(), -1e9), 1)
        a = torch.distributions.Categorical(logits=lp).sample()
        logPF += lp.gather(1, a[:, None]).squeeze(1)
        state = state.scatter(1, a[:, None], 1.0)
    idxs = np.array([set_to_idx[frozenset(np.where(state[b].numpy() > 0)[0].tolist())]
                     for b in range(B)])
    return idxs, logPF
def train_gflownet(reward, iters=1500):
    net = Policy(NCAS)
    opt = torch.optim.Adam([{"params": net.net.parameters(), "lr": 1e-3},
                            {"params": [net.logZ], "lr": 1e-1}])
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

# ---------- experiment: energy vs shots at fixed subspace dim ----------
D = 120
def topD_energy(counts):
    idxs = [i for i, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:D]
    strings = [strs_a[i] for i in idxs]
    ntrip = int((exc[idxs] >= 3).sum()); ntop = len(set(idxs) & top_true)
    return (E(strings) - e_fci) * 1000, len(idxs), ntrip, ntop

log(f"\n{'shots':>6s}  {'method':10s} {'Eerr(mHa)':>10s} {'dim':>4s} {'trip+':>5s} {'top60':>6s}")
cheap_sel = {i: w_cheap[i] for i in np.argsort(w_cheap)[::-1][:D]}   # shot-independent ceiling
for shots in (200, 500, 1000, 2000):
    bits = quantum_shots(shots)
    counts_raw = {}
    for b in bits:
        if int(b.sum()) == na:
            i = str_to_idx[bits_to_str(b)]; counts_raw[i] = counts_raw.get(i, 0) + 1
    occ = bits[[int(b.sum()) == na for b in bits]].mean(0) if len(counts_raw) else np.full(NCAS, na/NCAS)
    rec = recover_ibm(bits, occ)
    counts_ibm = {}
    for s in rec:
        i = str_to_idx[s]; counts_ibm[i] = counts_ibm.get(i, 0) + 1
    # GFlowNet reward: fuse recovered empirical freq (triples+ signal) + cheap (S+D prior)
    f_emp = np.zeros(dim_a)
    for i, c in counts_ibm.items(): f_emp[i] = c
    f_emp = f_emp / max(f_emp.sum(), 1)
    reward = f_emp + 0.1 * w_cheap / w_cheap.max() * f_emp.max()
    reward = np.maximum(reward, 1e-9)
    net = train_gflownet(reward)
    gcounts = {}
    for i in gfn_draws(net, 20000): gcounts[int(i)] = gcounts.get(int(i), 0) + 1

    for name, counts in [("raw", counts_raw), ("ibm", counts_ibm),
                         ("cheap", cheap_sel), ("gflownet", gcounts)]:
        e, d, tp, tt = topD_energy(counts)
        log(f"{shots:6d}  {name:10s} {e:10.2f} {d:4d} {tp:5d} {tt:4d}/{K}")
    log("")
log("Win condition: gflownet < raw and < ibm (esp. at low shots), recovering triples+ that")
log("cheap cannot -> shot-efficient, physically-grounded configuration recovery.")
log("DONE.")
