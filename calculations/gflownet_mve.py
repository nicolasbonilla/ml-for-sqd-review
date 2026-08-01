"""
GFlowNet-for-SQD — MVE core proof-of-concept
============================================
Implements a Generative Flow Network (Trajectory Balance objective, Malkin et al.
2022) that generates single-spin Slater determinants (fixed-electron-number
subsets of active orbitals) and learns to sample them in proportion to an
importance reward.

Mechanism claim we verify here (on N2/CAS ground truth):
  Given an importance signal, a GFlowNet builds a subspace that is BOTH
  (i) enriched in high-importance determinants (like importance sampling), and
  (ii) DIVERSE (unlike i.i.d. importance sampling, which wastes draws re-hitting
       the mode) -> better energy at fixed subspace dimension (compactness).

Baselines: uniform sampling; i.i.d. importance sampling ("oracle" with replacement).
Metric: subspace energy error vs number of UNIQUE determinants (= subspace dim),
following the compactness methodology of Reinholdt et al. (JCTC 2025).
"""
import time, math
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
import pyscf.fci
from pyscf.fci import cistring, selected_ci

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0)
rng = np.random.default_rng(0)

# ---------------- system + FCI ground truth (importance signal) ----------------
NCAS, NELECAS = 12, (5, 5)
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
na, nb = NELECAS
strs_a = cistring.make_strings(range(NCAS), na)
dim_a = len(strs_a)
idx_of = {int(s): i for i, s in enumerate(strs_a)}
fci = pyscf.fci.direct_spin1.FCI()
e_fci, civec = fci.kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_a)
w_a = (civec**2).sum(1); w_a = w_a / w_a.sum()          # importance signal
order = np.argsort(w_a)[::-1]
log(f"CAS({sum(NELECAS)}e,{NCAS}o)  {dim_a} strings  FCI={e_fci:.6f}")

_sci = selected_ci.SelectedCI()
def subspace_energy(strings):
    s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return float(out[0] if isinstance(out, (tuple, list)) else out)

# orbital indices occupied by each string (for state encoding / reward lookup)
def occ(s): return [p for p in range(NCAS) if (s >> p) & 1]
set_to_idx = {frozenset(occ(int(s))): i for i, s in enumerate(strs_a)}

# ---------------- GFlowNet (Trajectory Balance) ----------------
DEV = "cpu"
class Policy(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(n, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n))
        s.logZ = nn.Parameter(torch.zeros(1))
    def forward(s, state):                 # state: (B, n) binary
        return s.net(state)

net = Policy(NCAS).to(DEV)
opt = torch.optim.Adam([{"params": net.net.parameters(), "lr": 1e-3},
                        {"params": [net.logZ], "lr": 1e-1}])
BETA = 1.0                                  # reward exponent (sample ∝ w^BETA)
w_t = torch.tensor(np.maximum(w_a, 1e-10), dtype=torch.float32)

def sample_batch(B, greedy=False):
    """Generate B determinants by adding na orbitals sequentially. Returns idxs, logPF."""
    state = torch.zeros(B, NCAS, device=DEV)
    logPF = torch.zeros(B, device=DEV)
    for _ in range(na):
        logits = net(state)
        logits = logits.masked_fill(state.bool(), -1e9)     # cannot re-add occupied
        logp = torch.log_softmax(logits, dim=1)
        if greedy:
            a = logp.argmax(1)
        else:
            a = torch.distributions.Categorical(logits=logp).sample()
        logPF += logp.gather(1, a[:, None]).squeeze(1)
        state = state.scatter(1, a[:, None], 1.0)
    # map final states -> string indices
    idxs = np.array([set_to_idx[frozenset(np.where(state[b].numpy() > 0)[0].tolist())]
                     for b in range(B)])
    return idxs, logPF

log("training GFlowNet (Trajectory Balance)...")
B = 256
for it in range(4000):
    idxs, logPF = sample_batch(B)
    logR = BETA * torch.log(w_t[idxs])
    loss = ((net.logZ + logPF - logR) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 500 == 0:
        log(f"  it {it+1:4d}  loss {loss.item():8.4f}  logZ {net.logZ.item():7.3f}")

# ---------------- evaluation ----------------
@torch.no_grad()
def gflownet_draws(n):
    out = []
    while len(out) < n:
        idxs, _ = sample_batch(min(512, n - len(out)))
        out.extend(idxs.tolist())
    return np.array(out[:n])

# correctness: does empirical frequency track the reward?
big = gflownet_draws(20000)
freq = np.bincount(big, minlength=dim_a) / len(big)
mask = w_a > 1e-6
from numpy import corrcoef
pear = corrcoef(np.log(freq[mask] + 1e-9), np.log(w_a[mask]))[0, 1]
log(f"\nCorrectness: corr(log emp-freq, log reward) on important strings = {pear:.3f}")

# baselines
def uniform_draws(n): return rng.integers(0, dim_a, n)
def oracle_draws(n):  return rng.choice(dim_a, size=n, p=w_a)

K = 60; top = set(order[:K].tolist())
def curve(draws, checkpoints):
    seen = set(); disc = []; comp = []
    cps = sorted(checkpoints); j = 0
    for k, d in enumerate(draws):
        seen.add(int(d))
        if j < len(cps) and (k+1) == cps[j]:
            disc.append((k+1, len(seen & top), len(seen)))
            comp.append((len(seen), (subspace_energy([strs_a[i] for i in seen]) - e_fci)*1000))
            j += 1
    return disc, comp

CP = [50, 100, 200, 400, 800, 1600, 3200]
log(f"\n{'method':10s}{'draws':>7s}{'topK':>8s}{'uniq':>7s}{'Eerr(mHa)':>11s}")
res = {}
for name, dr in [("uniform", uniform_draws(max(CP))),
                 ("oracle", oracle_draws(max(CP))),
                 ("gflownet", gflownet_draws(max(CP)))]:
    disc, comp = curve(dr, CP); res[name] = comp
    for (d, f, u), (uu, e) in zip(disc, comp):
        log(f"{name:10s}{d:7d}{f:5d}/{K:<3d}{u:7d}{e:11.3f}")
    log("")

# headline: energy at matched subspace dimension (compactness)
log("=== COMPACTNESS: energy error (mHa) at matched subspace dimension ===")
for target in (30, 60, 120):
    line = f"  ~{target:4d} unique strings: "
    for name in ("uniform", "oracle", "gflownet"):
        arr = np.array(res[name])
        j = np.argmin(np.abs(arr[:, 0] - target))
        line += f"{name}={arr[j,1]:7.2f}  "
    log(line)
log("\nDONE.")
