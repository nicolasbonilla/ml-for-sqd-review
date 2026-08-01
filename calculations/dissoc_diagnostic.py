"""
Diagnostic: where does S+D-restricted selection FAIL?  (locating the GFlowNet's niche)
=====================================================================================
HCI-greedy on a cheap S+D criterion is near-optimal when triples+ are negligible.
We stretch N2 toward dissociation and measure, at each geometry:
  - TRUE weight carried by triples-or-higher (from HF),
  - best-possible compactness using ONLY S+D determinants (the ceiling of any
    cheap-S+D method), vs unrestricted best top-K (which may use triples+).
A large gap = a regime where a proposer that EXPLORES beyond S+D can genuinely win.
"""
import time
import numpy as np
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
NCAS, NELECAS = 12, (5, 5)
na, nb = NELECAS
strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
HF = (1 << na) - 1
exc = np.array([na - bin(int(s) & HF).count("1") for s in strs_a])
_sci = selected_ci.SelectedCI()

log(f"{'R(A)':>5s} {'FCI':>12s} {'trip+ wt%':>10s} {'E@120 all':>11s} {'E@120 S+D':>11s} {'gap(mHa)':>9s}")
for R in (2.0, 2.5, 3.0, 3.6):
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    cas = mcscf.CASCI(mf, NCAS, NELECAS)
    h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
    e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
    civec = civec.reshape(dim_a, dim_a); w = (civec**2).sum(1); w /= w.sum()
    trip = w[exc >= 3].sum()

    def Esel(strings):
        s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
        out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
        return float(out[0] if isinstance(out, (tuple, list)) else out)

    order = np.argsort(w)[::-1]
    top_all = [strs_a[i] for i in order[:120]]
    sd_order = [i for i in order if exc[i] <= 2][:120]           # best S+D-only subspace
    e_all = (Esel(top_all) - e_fci) * 1000
    e_sd = (Esel([strs_a[i] for i in sd_order]) - e_fci) * 1000
    log(f"{R:5.1f} {e_fci:12.6f} {100*trip:10.1f} {e_all:11.2f} {e_sd:11.2f} {e_sd-e_all:9.2f}")
log("\nLarger gap at long R => stronger case for a proposer that explores beyond S+D.")
