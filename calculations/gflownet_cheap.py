"""
GFlowNet-for-SQD — honest experiment with a CHEAP (FCI-free) reward
===================================================================
Removes the circularity of training on the FCI marginal. The reward is the
Epstein-Nesbet first-order (PT1) importance of each determinant w.r.t. the HF
reference -- exactly the cheap criterion CIPSI/HCI use to select determinants.
It needs ONE H|HF> matrix-vector product + the Hamiltonian diagonal, NO FCI.

We then benchmark, on the TRUE FCI energy (ground truth used ONLY for scoring):
  - uniform sampling                     (naive proxy)
  - i.i.d. sampling  ∝ cheap reward      (importance sampling)
  - DETERMINISTIC top-K by cheap reward  (= HCI/CIPSI greedy selection: the
                                            strongest classical competitor)
  - GFlowNet trained on the cheap reward (ours)
  - oracle (FCI marginal)                (unreachable upper bound)
Metric: subspace energy error vs subspace dimension (compactness, Reinholdt-style).
"""
import time
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)

# ---------------- system ----------------
NCAS, NELECAS = 12, (5, 5)
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
na, nb = NELECAS
strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
set_to_idx = {frozenset(p for p in range(NCAS) if (int(s) >> p) & 1): i
              for i, s in enumerate(strs_a)}
hf_idx = int(np.where(strs_a == (1 << na) - 1)[0][0])
log(f"CAS({sum(NELECAS)}e,{NCAS}o)  {dim_a} strings  HF-string idx={hf_idx}")

# ---------------- CHEAP reward: Epstein-Nesbet PT1 (no FCI) ----------------
civ_hf = np.zeros((dim_a, dim_a)); civ_hf[hf_idx, hf_idx] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ_hf, NCAS, NELECAS).reshape(dim_a, dim_a)  # <D|H|HF>
hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)     # <D|H|D>
E_hf = Hc[hf_idx, hf_idx]
denom = E_hf - hdiag; denom[hf_idx, hf_idx] = 1.0
c1 = Hc / denom; c1[hf_idx, hf_idx] = 1.0                    # PT1 coefficients
w_cheap = (c1**2).sum(axis=1); w_cheap = w_cheap / w_cheap.sum()
log("cheap Epstein-Nesbet PT1 reward computed (1 matvec, no FCI)")

# ---------------- FCI ground truth (ONLY for scoring) ----------------
e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_a)
w_true = (civec**2).sum(1); w_true = w_true / w_true.sum()
order_true = np.argsort(w_true)[::-1]
sp = np.corrcoef(np.log(w_cheap + 1e-12), np.log(w_true + 1e-12))[0, 1]
# how well does cheap ranking recover the true top-60?
K = 60; top_true = set(order_true[:K].tolist())
top_cheap = set(np.argsort(w_cheap)[::-1][:K].tolist())
log(f"FCI={e_fci:.6f}. cheap-vs-true log-corr={sp:.3f}; "
    f"cheap top-{K} recovers {len(top_true & top_cheap)}/{K} of true top-{K}")

_sci = selected_ci.SelectedCI()
def E(strings):
    s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return float(out[0] if isinstance(out, (tuple, list)) else out)

# ---------------- GFlowNet (Trajectory Balance) on the CHEAP reward ----------------
class Policy(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 256),
                              nn.ReLU(), nn.Linear(256, n))
        s.logZ = nn.Parameter(torch.zeros(1))
    def forward(s, x): return s.net(x)
net = Policy(NCAS)
opt = torch.optim.Adam([{"params": net.net.parameters(), "lr": 1e-3},
                        {"params": [net.logZ], "lr": 1e-1}])
w_r = torch.tensor(np.maximum(w_cheap, 1e-10), dtype=torch.float32)

def sample_batch(B):
    state = torch.zeros(B, NCAS); logPF = torch.zeros(B)
    for _ in range(na):
        lg = net(state).masked_fill(state.bool(), -1e9)
        lp = torch.log_softmax(lg, 1)
        a = torch.distributions.Categorical(logits=lp).sample()
        logPF += lp.gather(1, a[:, None]).squeeze(1)
        state = state.scatter(1, a[:, None], 1.0)
    idxs = np.array([set_to_idx[frozenset(np.where(state[b].numpy() > 0)[0].tolist())]
                     for b in range(B)])
    return idxs, logPF

log("training GFlowNet on cheap reward...")
for it in range(3000):
    idxs, logPF = sample_batch(256)
    loss = ((net.logZ + logPF - torch.log(w_r[idxs])) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 1000 == 0: log(f"  it {it+1} loss {loss.item():.4f}")

@torch.no_grad()
def gfn_draws(n):
    o = []
    while len(o) < n:
        idxs, _ = sample_batch(min(512, n - len(o))); o += idxs.tolist()
    return np.array(o[:n])

# ---------------- baselines + evaluation (compactness on TRUE energy) ----------------
def uniform_draws(n): return rng.integers(0, dim_a, n)
def cheap_iid_draws(n): return rng.choice(dim_a, size=n, p=w_cheap)
def det_topk(m):        return list(np.argsort(w_cheap)[::-1][:m])   # HCI/CIPSI greedy

SIZES = [30, 60, 120, 240]
def compactness(draws, sizes):
    """Record (unique, energy_err_mHa) the first time #unique reaches each size."""
    seen, out, sset, j = set(), {}, sorted(sizes), 0
    for d in draws:
        seen.add(int(d))
        if j < len(sset) and len(seen) >= sset[j]:
            out[sset[j]] = (len(seen), (E([strs_a[i] for i in seen]) - e_fci) * 1000); j += 1
    return out, len(seen)

methods, maxuniq = {}, {}
for name, dr in [("uniform", uniform_draws(8000)),
                 ("cheap-iid", cheap_iid_draws(50000)),
                 ("gflownet", gfn_draws(50000)),
                 ("oracle", rng.choice(dim_a, 50000, p=w_true))]:
    methods[name], maxuniq[name] = compactness(dr, SIZES)
# deterministic greedy selection (HCI/CIPSI) at exact sizes — always reaches m
det = {m: (E([strs_a[i] for i in det_topk(m)]) - e_fci) * 1000 for m in SIZES}

log("\nmax unique strings reached: " +
    "  ".join(f"{k}={v}" for k, v in maxuniq.items()) + f"   (i.i.d. reveals collapse)")
log("\n=== COMPACTNESS: true energy error (mHa) vs subspace dimension ===")
log(f"{'dim':>5s} {'uniform':>9s} {'cheap-iid':>10s} {'HCI-greedy':>11s} {'GFlowNet':>10s} {'oracle':>9s}")
cell = lambda name, m: f"{methods[name][m][1]:.2f}" if m in methods[name] else "  --"
for m in SIZES:
    log(f"{m:5d} {cell('uniform',m):>9s} {cell('cheap-iid',m):>10s} "
        f"{det[m]:11.2f} {cell('gflownet',m):>10s} {cell('oracle',m):>9s}")
log("\nInterpretation: HCI-greedy = strongest classical competitor (deterministic top-K")
log("by the SAME cheap criterion). GFlowNet vs HCI-greedy at equal dim is the honest test.")
log("DONE.")
