"""
GFlowNet-for-SQD — the contribution: REWARD TEMPERING for diverse, compact subspaces
====================================================================================
Finding from gflownet_cheap.py: sampling ∝ a peaked importance reward (β=1) mode-
collapses (14 unique strings). i.i.d. importance sampling collapses too (33).
Deterministic HCI-greedy is the strong classical competitor but is BLIND to
determinants with zero cheap-reward (triples+ from HF), which matter under strong
correlation.

Hypothesis: a GFlowNet with a TEMPERED reward R = (w_cheap ⊕ floor)^β (β<1) trades
importance for diversity, reaches large subspace dimension, and can EXPLORE beyond
the S+D support of the cheap criterion -> beating HCI-greedy at equal dimension.

We sweep β, and for each method report (i) max unique reached, (ii) compactness on
the TRUE FCI energy, and (iii) the fraction of selected determinants that are
triples-or-higher from HF (the part greedy-on-cheap can never reach).
"""
import time
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0); rng = np.random.default_rng(0)

NCAS, NELECAS = 12, (5, 5)
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
na, nb = NELECAS
strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
set_to_idx = {frozenset(p for p in range(NCAS) if (int(s) >> p) & 1): i
              for i, s in enumerate(strs_a)}
HF = (1 << na) - 1; hf_idx = int(np.where(strs_a == HF)[0][0])
exc = np.array([na - bin(int(s) & HF).count("1") for s in strs_a])   # excitation level from HF

# cheap Epstein-Nesbet PT1 reward (no FCI)
civ_hf = np.zeros((dim_a, dim_a)); civ_hf[hf_idx, hf_idx] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ_hf, NCAS, NELECAS).reshape(dim_a, dim_a)
hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)
denom = Hc[hf_idx, hf_idx] - hdiag; denom[hf_idx, hf_idx] = 1.0
c1 = Hc / denom; c1[hf_idx, hf_idx] = 1.0
w_cheap = (c1**2).sum(1); w_cheap = w_cheap / w_cheap.sum()
frac_sd = (w_cheap[exc <= 2].sum())
log(f"{dim_a} strings. cheap reward support: {(w_cheap>1e-12).sum()} strings; "
    f"{100*frac_sd:.1f}% of cheap weight is S+D; triples+ get ZERO cheap reward")

# FCI ground truth (scoring only)
e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_a); w_true = (civec**2).sum(1); w_true /= w_true.sum()
imp_triples = w_true[exc >= 3].sum()
log(f"FCI={e_fci:.6f}. TRUE weight in triples+ = {100*imp_triples:.1f}%  "
    f"(this is what HCI-greedy-on-cheap structurally misses)")

_sci = selected_ci.SelectedCI()
def E(strings):
    s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return float(out[0] if isinstance(out, (tuple, list)) else out)

# ---------- GFlowNet with tempered reward ----------
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

FLOOR = 1e-3 * w_cheap.max()          # gives triples+ a small nonzero reward -> explorable
def train_gflownet(beta, iters=2000):
    net = Policy(NCAS)
    opt = torch.optim.Adam([{"params": net.net.parameters(), "lr": 1e-3},
                            {"params": [net.logZ], "lr": 1e-1}])
    R = np.maximum(w_cheap, FLOOR) ** beta
    R = torch.tensor(R / R.sum(), dtype=torch.float32)
    for it in range(iters):
        idxs, logPF = sample_batch(net, 256)
        loss = ((net.logZ + logPF - torch.log(R[idxs])) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

@torch.no_grad()
def gfn_draws(net, n):
    o = []
    while len(o) < n:
        idxs, _ = sample_batch(net, 512); o += idxs.tolist()
    return np.array(o[:n])

# ---------- evaluation ----------
SIZES = [30, 60, 120, 240]
def compactness(draws):
    seen, out, sset, j = set(), {}, sorted(SIZES), 0
    trip = 0
    for d in draws:
        seen.add(int(d))
        if j < len(sset) and len(seen) >= sset[j]:
            sel = [strs_a[i] for i in seen]
            ntrip = int((exc[list(seen)] >= 3).sum())
            out[sset[j]] = ((E(sel) - e_fci) * 1000, 100 * ntrip / len(seen)); j += 1
    return out, len(seen)

def det_topk(m): return list(np.argsort(w_cheap)[::-1][:m])
det = {}
for m in SIZES:
    sel = det_topk(m); ntrip = int((exc[sel] >= 3).sum())
    det[m] = ((E([strs_a[i] for i in sel]) - e_fci) * 1000, 100 * ntrip / m)

rows = {}
rows["HCI-greedy"] = (det, m)
for beta in (0.3, 0.5, 1.0):
    log(f"training GFlowNet  β={beta} ...")
    net = train_gflownet(beta)
    comp, mu = compactness(gfn_draws(net, 60000))
    rows[f"GFN β={beta}"] = (comp, mu)
    log(f"   max unique = {mu}")

log("\n=== COMPACTNESS on TRUE FCI energy — error mHa  (triples+ % of subspace) ===")
hdr = f"{'method':14s}" + "".join(f"{'dim '+str(m):>16s}" for m in SIZES)
log(hdr)
def fmt(name, m):
    d = rows[name][0]
    if name == "HCI-greedy": e, tp = det[m]
    elif m in d: e, tp = d[m]
    else: return f"{'--':>16s}"
    return f"{e:8.2f} ({tp:3.0f}%) "
for name in ["HCI-greedy", "GFN β=1.0", "GFN β=0.5", "GFN β=0.3"]:
    log(f"{name:14s}" + "".join(fmt(name, m) for m in SIZES))
log("\nRead: lower mHa = better. (x%) = share of subspace that is triples+ (structurally")
log("unreachable by greedy-on-cheap). If a tempered GFN matches/beats HCI-greedy while")
log("carrying triples+, that is the physically-grounded contribution.")
log("DONE.")
