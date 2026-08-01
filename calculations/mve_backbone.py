"""
GFlowNet-for-SQD — MVE measurement apparatus  (v2, methodologically correct)
============================================================================
Key correction over v1: the SQD subspace is the PRODUCT of the unique single-spin
strings seen (alpha x beta), exactly as in build_subspace (QGSS Lab 4c). So the
coupon-collector problem lives at the level of SINGLE-SPIN STRINGS, and proposers
must be evaluated on how efficiently they DISCOVER the important strings.

We therefore:
  - use an active space with MANY single-spin strings (so the effect is visible),
  - define the reward/importance at the single-spin-string level (FCI marginal),
  - measure (i) string-discovery curves and (ii) subspace energy vs # strings.

Outputs a verified apparatus + saves arrays to mve_data.npz + a figure.
"""
import time
import numpy as np
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
import pyscf.fci
from pyscf.fci import cistring, selected_ci

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- system
R = 2.0
NCAS = 12
NELECAS = (5, 5)                      # C(12,5)=792 single-spin strings; FCI dim 792^2=627k
mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas()
h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
na, nb = NELECAS
strs_a = cistring.make_strings(range(NCAS), na)
strs_b = cistring.make_strings(range(NCAS), nb)
dim_a, dim_b = len(strs_a), len(strs_b)
log(f"N2 @ {R}A  CAS({sum(NELECAS)}e,{NCAS}o)  HF={mf.e_tot:.6f}")
log(f"single-spin strings: {dim_a}   FCI dim: {dim_a*dim_b}")

# ---------------------------------------------------------------- FCI ground truth
fci = pyscf.fci.direct_spin1.FCI()
e_fci, civec = fci.kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_b)
log(f"FCI = {e_fci:.6f}")

# marginal single-spin string weights  w_a(i) = sum_b |c_{i,b}|^2  (== w_b by symmetry)
w_a = (civec**2).sum(axis=1)
w_a = w_a / w_a.sum()
order = np.argsort(w_a)[::-1]
cum = np.cumsum(w_a[order])
log("Coupon-collector signature (over single-spin strings):")
for frac in (0.90, 0.99, 0.999):
    k = int(np.searchsorted(cum, frac)) + 1
    log(f"   {frac*100:5.1f}% of marginal weight in top {k:4d}/{dim_a} strings ({100*k/dim_a:.1f}%)")

# ---------------------------------------------------------------- subspace energy
_sci = selected_ci.SelectedCI()
def subspace_energy(strings):
    s = np.asarray(sorted(strings), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    e = out[0] if isinstance(out, (tuple, list)) else out
    return float(e)

assert abs(subspace_energy(strs_a) - e_fci) < 1e-6
log("Sanity: full single-spin space == FCI  ✓")

# ---------------------------------------------------------------- proposers (over strings)
def prop_uniform(n):
    return rng.integers(0, dim_a, n)
def prop_oracle(n):                      # sample strings ∝ FCI marginal (unreachable UB)
    return rng.choice(dim_a, size=n, p=w_a)

# ---------------------------------------------------------------- metrics
K = 60
top_strings = set(order[:K].tolist())
def evaluate(prop, n_draws, checkpoints):
    draws = prop(n_draws)
    seen = set(); found_c = []; en_c = []
    cps = sorted(checkpoints); j = 0
    for k in range(n_draws):
        seen.add(int(draws[k]))
        if j < len(cps) and (k+1) == cps[j]:
            found_c.append((k+1, len(seen & top_strings), len(seen)))
            en_c.append((k+1, (subspace_energy([strs_a[i] for i in seen]) - e_fci)*1000))
            j += 1
    return found_c, en_c

checkpoints = [50, 100, 200, 400, 800, 1600, 3200]
log(f"\n{'method':10s} {'draws':>6s} {'topK':>7s} {'uniq':>6s} {'Eerr(mHa)':>10s}")
data = {}
for name, prop in [("uniform", prop_uniform), ("oracle", prop_oracle)]:
    fc, ec = evaluate(prop, max(checkpoints), checkpoints)
    data[name] = (fc, ec)
    for (d, f, u), (_, e) in zip(fc, ec):
        log(f"{name:10s} {d:6d} {f:3d}/{K:<3d} {u:6d} {e:10.3f}")
    log("")

np.savez("mve_data.npz", w_a=w_a, order=order, e_fci=e_fci,
         uniform=np.array(data["uniform"][1]), oracle=np.array(data["oracle"][1]),
         uniform_found=np.array(data["uniform"][0]), oracle_found=np.array(data["oracle"][0]))
log("saved mve_data.npz")
print("\n=== APPARATUS VERIFIED ===")
print("The gap between 'uniform' and 'oracle' at fixed #draws is exactly the room")
print("the GFlowNet must capture — discovering important strings from a cheap reward.")
