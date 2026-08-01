"""
Strong classical baseline: iterative Heat-bath CI (HCI) over single-spin strings.
Answers the Reinholdt question: does the quantum+generative approach ever beat a
strong PURELY CLASSICAL selected-CI at matched subspace dimension (same product
structure)? HCI bootstraps from the current correlated wavefunction (NOT HF), so it
handles multireference C2 well. Purely classical (no quantum samples, noise-free).

Reports, per molecule, at dim D=120 strings:  HCI  vs  oracle(top-D by true weight)
vs the GFlowNet numbers already measured. If HCI <= gfn everywhere, classical wins
at verifiable scale (the honest Reinholdt verdict).
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
MOLS = {
    "h2o": dict(atom="O 0 0 0; H 0 0.98 0.76; H 0 -0.98 0.76", basis="cc-pvdz", ncore=1, ncas=12, nelecas=(4,4), gfn=2.1),
    "n2":  dict(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", ncore=2, ncas=12, nelecas=(5,5), gfn=26.7),
    "c2":  dict(atom="C 0 0 0; C 0 0 1.40", basis="cc-pvdz", ncore=2, ncas=12, nelecas=(4,4), gfn=9.7),
}
D = 120

for name, S in MOLS.items():
    NCAS, NELECAS = S["ncas"], S["nelecas"]; na, nb = NELECAS
    mol = gto.M(atom=S["atom"], basis=S["basis"], verbose=0)
    mf = scf.RHF(mol).run()
    cas = mcscf.CASCI(mf, NCAS, NELECAS); cas.ncore = S["ncore"]
    h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
    strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
    HF = (1 << na) - 1; hf_idx = int(np.where(strs_a == HF)[0][0])
    h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
    hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)
    e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
    civec = civec.reshape(dim_a, dim_a); w_true = (civec**2).sum(1)

    def subspace_ground(sA):
        idx = np.asarray(sorted(set(sA))); n = len(idx)
        def mv(x):
            full = np.zeros((dim_a, dim_a)); full[np.ix_(idx, idx)] = x.reshape(n, n)
            Hf = direct_spin1.contract_2e(h2e, full, NCAS, NELECAS).reshape(dim_a, dim_a)
            return Hf[np.ix_(idx, idx)].ravel()
        if n == 1:
            full = np.zeros((dim_a, dim_a)); full[idx[0], idx[0]] = 1.0
            e = direct_spin1.contract_2e(h2e, full, NCAS, NELECAS).reshape(dim_a, dim_a)[idx[0], idx[0]]
            return float(e), np.array([[1.0]]), idx
        op = LinearOperator((n*n, n*n), matvec=mv)
        w, v = eigsh(op, k=1, which="SA", maxiter=2000, tol=1e-7)
        return float(w[0]), v[:, 0].reshape(n, n), idx

    # iterative HCI: grow subspace in batches, scoring by EN-PT from current wavefunction
    A = {hf_idx}; batch = max(12, D // 8)
    while len(A) < D:
        e_el, c, idx = subspace_ground(A)
        full = np.zeros((dim_a, dim_a)); full[np.ix_(idx, idx)] = c
        Hc = direct_spin1.contract_2e(h2e, full, NCAS, NELECAS).reshape(dim_a, dim_a)
        gap = e_el - hdiag; gap = np.where(np.abs(gap) < 1e-6, 1e-6, gap)
        score = ((Hc**2) / (gap**2)).sum(axis=1)
        score[list(A)] = -np.inf
        need = min(batch, D - len(A))
        A.update(np.argsort(score)[::-1][:need].tolist())
    e_hci, _, _ = subspace_ground(A)
    E_hci = (e_hci + ecore - e_fci) * 1000

    oracle = np.argsort(w_true)[::-1][:D]
    e_or, _, _ = subspace_ground(oracle)
    E_or = (e_or + ecore - e_fci) * 1000

    log(f"[{name:4s}] dim={D}  HCI(classical)={E_hci:7.2f} mHa   oracle(top-D)={E_or:7.2f}   "
        f"GFlowNet(noisy)~{S['gfn']:.1f}   -> {'CLASSICAL wins' if E_hci <= S['gfn'] else 'GFN competitive'}")
log("\nVerdict: if HCI <= GFlowNet for all molecules, classical selected-CI wins at")
log("FCI-verifiable scale (honest Reinholdt result). Quantum+generative value would then")
log("lie only at scales beyond FCI truth (the scale-vs-verifiability tension).")
