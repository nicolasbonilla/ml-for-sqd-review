"""
Strong classical baseline: iterative EPSTEIN-NESBET selected CI over single-spin strings.

NOMBRE. Este script se llamo 'hci_baseline' y su docstring decia 'Heat-bath CI', y
ninguna de las dos cosas es exacta: el criterio de seleccion de abajo puntua por
(H.c)^2/Delta^2, que es Epstein-Nesbet. Lo que define a HCI es el cribado |H_ai c_i|,
que no esta aqui. El manuscrito siempre lo llamo bien -- 'EN-selected CI, CIPSI-style'
-- asi que el que estaba mal era el deposito. El nombre del fichero se conserva para
no romper las rutas que el README y el manuscrito ya citan; el metodo es el de arriba.

Y ES UNA IMPLEMENTACION PROPIA, no un codigo publicado. La seccion 5.1 del paper pide
que un benchmark nombre el codigo contra el que se corrio; el nuestro no corre contra
Dice, Quantum Package, block2, ipie ni NECI, sino contra estas ~35 lineas.
Answers the Reinholdt question: does the quantum+generative approach ever beat a
strong PURELY CLASSICAL selected-CI at matched subspace dimension (same product
structure)? HCI bootstraps from the current correlated wavefunction (NOT HF), so it
handles multireference C2 well. Purely classical (no quantum samples, noise-free).

Reports, per molecule, at dim D=120 strings:  HCI  vs  oracle(top-D by true weight)
vs the GFlowNet numbers already measured. If HCI <= gfn everywhere, classical wins
at verifiable scale (the honest Reinholdt verdict).

GAUGE (see gauge_study/). Every molecule is built with symmetry=True. This comparison is
only fair if both sides rank strings in the SAME basis, and canonical RHF orbitals do not
provide one: left unfixed they rotate freely inside the degenerate pi shells (N2, C2) at
identical energy to 1e-13 Ha, moving the string weights and hence the verdict. Fixing the
point group makes the comparison a property of the method, not of the run. Run with
OMP_NUM_THREADS=1.
"""
import time
import json
import os

import numpy as np

SALIDA = {}   # se vuelca a results/hci_baseline.json al final

def _lee_gfn():
    """Las barras generativas, del fichero que las mide. Si no esta, se dice y se sigue."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                     "results", "gfn_ruidoso_fig7.json")
    try:
        d = json.load(open(p, encoding="utf-8"))
        return {k: round(v["gflownet"]["mean"], 2) for k, v in d["mols"].items()}
    except Exception as e:
        print("AVISO: no puedo leer gfn_ruidoso_fig7.json (%s); las barras generativas"
              " quedan a None y la comparacion final no se imprime." % e)
        return {}

_GFN = _lee_gfn()
# c2 no tiene medida propia: gfn_ruidoso_fig7.py corre las dos moleculas que la Fig. 7
# usa, y el paper omite C2 de esa figura porque su referencia FCI no es exacta aqui.
# Se deja explicitamente sin medir, en vez de arrastrar el literal 9.7 que habia.
_GFN.setdefault("h2o", None); _GFN.setdefault("n2", None); _GFN.setdefault("c2", None)
from scipy.sparse.linalg import LinearOperator, eigsh
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
MOLS = {
    # gfn: error del proponente generativo bajo ruido Heron a D=120. NO lo calcula este
    # script; se LEE de results/gfn_ruidoso_fig7.json, que lo mide sobre cinco semillas.
    # Hasta el 2026-09-14 estaba escrito a mano aqui como 2.1 y 26.7, y ningun script del
    # deposito lo producia: era la mitad no comprobable de la figura decisiva del paper.
    "h2o": dict(atom="O 0 0 0; H 0 0.98 0.76; H 0 -0.98 0.76", basis="cc-pvdz", ncore=1, ncas=12, nelecas=(4,4), gfn=_GFN["h2o"]),
    "n2":  dict(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", ncore=2, ncas=12, nelecas=(5,5), gfn=_GFN["n2"]),
    "c2":  dict(atom="C 0 0 0; C 0 0 1.40", basis="cc-pvdz", ncore=2, ncas=12, nelecas=(4,4), gfn=_GFN["c2"]),
}
D = 120

for name, S in MOLS.items():
    NCAS, NELECAS = S["ncas"], S["nelecas"]; na, nb = NELECAS
    mol = gto.M(atom=S["atom"], basis=S["basis"], symmetry=True, verbose=0)
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
        f"GFlowNet(noisy)~{'sin medir' if S['gfn'] is None else format(S['gfn'], '.1f')}   -> "
        f"{'no comparable' if S['gfn'] is None else ('CLASSICAL wins' if E_hci <= S['gfn'] else 'GFN competitive')}")
    # Se DEPOSITA. Hasta el 2026-09-14 este script no escribia nada, asi que sus numeros
    # no se podian contrastar con nada -- y el manuscrito llego a imprimir 9.9 mHa para
    # N2 donde este generador da 9.13, porque la cifra se habia "reconciliado" a mano en
    # figures/hci_fig.py. Ahora la figura lee este fichero.
    # DIAGNOSTICO. hci_mHa es un error contra la verdad exacta, y el principio
    # variacional lo hace no negativo: un subespacio de la FCI no puede quedar por debajo
    # de la FCI. Si sale negativo, la referencia no es exacta -- en C2 el FCI de este
    # espacio activo es casi degenerado y el solucionador disperso llega a caer por
    # debajo de la raiz reportada. Se marca en el fichero en vez de dejar el numero solo.
    fiable = bool(E_hci >= -1e-6 and E_or >= -1e-6)
    SALIDA[name] = {"dim": D, "hci_mHa": round(float(E_hci), 4),
                    "oracle_mHa": round(float(E_or), 4), "gfn_mHa": S["gfn"],
                    "E_FCI_Ha": float(e_fci), "atom": S["atom"], "basis": S["basis"],
                    "ncore": S["ncore"], "ncas": S["ncas"], "nelecas": list(S["nelecas"]),
                    "referencia_exacta_fiable": fiable}
    if not fiable:
        SALIDA[name]["aviso"] = (
            "Error NEGATIVO contra la referencia: el principio variacional lo prohibe, "
            "asi que la referencia FCI de este sistema no es exacta. El espacio activo es "
            "casi degenerado y el solucionador disperso cae por debajo de la raiz "
            "reportada. NO usar estos dos numeros; el paper omite este sistema de la "
            "Fig. 7 por esta razon.")
        log(f"   AVISO: {name} da un error negativo -- la referencia no es exacta. "
            f"Marcado en el fichero y excluido de la Fig. 7.")
log("\nVerdict: if HCI <= GFlowNet for all molecules, classical selected-CI wins at")
log("FCI-verifiable scale (honest Reinholdt result). Quantum+generative value would then")
log("lie only at scales beyond FCI truth (the scale-vs-verifiability tension).")

_DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                     "results", "hci_baseline.json")
SALIDA["_nota"] = ("gauge D-infinity-h adaptado por simetria, OMP_NUM_THREADS=1. "
                   "gfn_mHa es una ENTRADA, no algo que este script calcule: es el error "
                   "del proponente generativo bajo ruido, medido sobre cinco semillas por "
                   "calculations/gfn_ruidoso_fig7.py. Para c2 vale null: esa molecula no "
                   "esta en la Fig. 7 y su referencia FCI no es exacta aqui.")
with open(_DEST, "w", encoding="utf-8") as _fh:
    json.dump(SALIDA, _fh, indent=1, ensure_ascii=False)
    _fh.write("\n")
log("WROTE %s" % os.path.normpath(_DEST))
