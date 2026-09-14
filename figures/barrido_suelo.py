# -*- coding: utf-8 -*-
# RECUPERADO 2026-09-10 de la transcripcion de sesion y depositado aqui.
#
# Este es el script que produjo la Fig. 6 del paper. El que estaba depositado
# (compact_fig.py) NO la reproduce: usa np.maximum(w_cheap, 1e-12) mientras que
# la figura se hizo con FLOOR = 1e-3 * w_cheap.max(), nueve ordenes de magnitud
# mas alto. Con el suelo equivocado el GFlowNet da ~19 mHa a D=120 en vez de
# ~110.8 +- 11.0 con el suelo de produccion. El suelo mueve la MAGNITUD, no quien
# gana: el barrido de abajo lo establece, y el selector codicioso gana a los tres
# suelos que pueden llenar D=120.
#
# El suelo NO es un detalle de implementacion: es el "suelo explorable" que
# fuerza al muestreador a explorar. Con 546 de las 792 cadenas en recompensa
# exactamente cero (las triples-y-superiores, por Slater-Condon; Brillouin NO
# aplica al nivel de cadena alpha, vease la seccion 7 del paper), el suelo decide
# cuanta masa reciben esas cadenas y por tanto decide el resultado.
#
# CIFRAS DE ESTE SCRIPT. Las de produccion son las que emite el propio script con
# FLOOR_REL=1e-3 y estan depositadas en ../results/fig6_suelo_1e-03.json:
#   D=120  greedy 46.41 | gfn 110.8+-11.0 | uniform 492.4 | oracle 16.11+-1.65
# Son las que imprime la seccion 4.5 del paper.
#
# No confundirlas con la verificacion del 2026-09-10 (3 semillas, gfn D=120 167+-37,
# greedy 41.37), que se hizo ANTES de fijar el gauge con symmetry=True. Esa corrida
# rankeaba un subespacio distinto al mismo E_FCI y sus numeros estan retirados; se
# mencionan solo para que quien los haya leido sepa por que no coinciden.
#
# EL SUELO NO PONE NADA A CERO. Se aplica como np.maximum(w_cheap, FLOOR), de modo que
# las 792 cadenas conservan probabilidad no nula. Que a 1e-12 no se llegue a D=120 es
# una afirmacion sobre las 5e4 muestras por brazo, no sobre el soporte.
"""Fig 6 compactness with 5-seed error bars (uniform, GFlowNet, oracle stochastic; greedy deterministic).

GAUGE (see gauge_study/). mol.symmetry=True fixes the orbital gauge. The five seeds below are
meant to measure the spread of the SAMPLER; with canonical RHF orbitals left free to rotate
inside the degenerate pi shells, the subspace being ranked also changes from run to run at
identical energy (1e-13 Ha), so the error bars would silently mix sampler noise with gauge
noise. Run with OMP_NUM_THREADS=1."""
import time, json
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

NCAS, NELECAS = 12, (5, 5)
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
na, nb = NELECAS
strs_a = cistring.make_strings(range(NCAS), na); dim_a = len(strs_a)
set_to_idx = {frozenset(p for p in range(NCAS) if (int(s) >> p) & 1): i for i, s in enumerate(strs_a)}
hf_idx = int(np.where(strs_a == (1 << na) - 1)[0][0])
civ_hf = np.zeros((dim_a, dim_a)); civ_hf[hf_idx, hf_idx] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ_hf, NCAS, NELECAS).reshape(dim_a, dim_a)
hdiag = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim_a, dim_a)
E_hf = Hc[hf_idx, hf_idx]; denom = E_hf - hdiag; denom[hf_idx, hf_idx] = 1.0
c1 = Hc / denom; c1[hf_idx, hf_idx] = 1.0
w_cheap = (c1**2).sum(axis=1); w_cheap = w_cheap / w_cheap.sum()
e_fci, civec = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
civec = civec.reshape(dim_a, dim_a); w_true = (civec**2).sum(1); w_true /= w_true.sum()
log(f"FCI={e_fci:.6f}")
_sci = selected_ci.SelectedCI()
def E(strings):
    s = np.asarray(sorted(set(int(x) for x in strings)), dtype=np.int64)
    out = selected_ci.kernel_fixed_space(_sci, h1, h2, NCAS, NELECAS, (s, s), ecore=ecore)
    return float(out[0] if isinstance(out, (tuple, list)) else out)
SIZES = [30, 60, 120, 240]
def compactness(draws, sizes):
    seen, out, sset, j = set(), {}, sorted(sizes), 0
    for d in draws:
        seen.add(int(d))
        if j < len(sset) and len(seen) >= sset[j]:
            out[sset[j]] = (E([strs_a[i] for i in seen]) - e_fci) * 1000; j += 1
    return out
# deterministic greedy top-K (HCI/CIPSI) — seed-independent
det_topk = lambda m: list(np.argsort(w_cheap)[::-1][:m])
det = {m: (E([strs_a[i] for i in det_topk(m)]) - e_fci) * 1000 for m in SIZES}
log("greedy(top-K) deterministic: " + str({k: round(v,2) for k,v in det.items()}))

class Policy(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(n,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,n))
        s.logZ = nn.Parameter(torch.zeros(1))
    def forward(s, x): return s.net(x)
import os
# BARRIDO DEL SUELO. Un solo ajuste del suelo deja la comparacion sin resolver, asi
# que se corren tres (1e-3, 1e-6 y 1e-12 del maximo), cinco semillas cada uno, todo lo
# demas fijo, y se publican los tres. Resultado a D=120: 110.8+-11.0, 58.2+-6.6 y
# ningun valor, contra 46.41 mHa del selector codicioso determinista. Bajar el suelo
# mejora al proponente pero no le hace adelantar a esa dimension.
FLOOR_REL = float(os.environ.get("FLOOR_REL", "1e-3"))
FLOOR = FLOOR_REL * w_cheap.max()
w_r = torch.tensor(np.maximum(w_cheap, FLOOR) ** 0.5, dtype=torch.float32)

acc = {m:{"uniform":[], "gflownet":[], "oracle":[]} for m in SIZES}
for seed in range(5):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    net = Policy(NCAS)
    opt = torch.optim.Adam([{"params": net.net.parameters(), "lr":1e-3}, {"params":[net.logZ], "lr":1e-1}])
    def sample_batch(B):
        state = torch.zeros(B, NCAS); logPF = torch.zeros(B)
        for _ in range(na):
            lg = net(state).masked_fill(state.bool(), -1e9); lp = torch.log_softmax(lg,1)
            a = torch.distributions.Categorical(logits=lp).sample()
            logPF += lp.gather(1, a[:,None]).squeeze(1); state = state.scatter(1, a[:,None], 1.0)
        idxs = np.array([set_to_idx[frozenset(np.where(state[b].numpy()>0)[0].tolist())] for b in range(B)])
        return idxs, logPF
    for it in range(2000):
        idxs, logPF = sample_batch(256)
        loss = ((net.logZ + logPF - torch.log(w_r[idxs]))**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def gfn_draws(n):
        o=[]
        while len(o)<n:
            idxs,_=sample_batch(min(512,n-len(o))); o+=idxs.tolist()
        return np.array(o[:n])
    cu = compactness(rng.integers(0, dim_a, 8000), SIZES)
    cg = compactness(gfn_draws(50000), SIZES)
    co = compactness(rng.choice(dim_a, 50000, p=w_true), SIZES)
    for m in SIZES:
        if m in cu: acc[m]["uniform"].append(cu[m])
        if m in cg: acc[m]["gflownet"].append(cg[m])
        if m in co: acc[m]["oracle"].append(co[m])
    log(f"seed {seed}: uniform={ {k:round(v,1) for k,v in cu.items()} } gfn={ {k:round(v,1) for k,v in cg.items()} }")

res = {"det_greedy": det, "fci": e_fci, "SIZES": SIZES, "stats": {}}
for m in SIZES:
    res["stats"][m] = {}
    for name in ("uniform","gflownet","oracle"):
        v = acc[m][name]
        if v: res["stats"][m][name] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1) if len(v)>1 else 0.0), "n": len(v)}
# El nombre del fichero sale de la MISMA variable que el suelo, con el mismo valor
# por defecto: antes esta linea leia os.environ["FLOOR_REL"] sin defecto y lanzaba
# KeyError despues de entrenar las cinco semillas, tirando la corrida entera.
# El nombre se normaliza a la forma que el deposito usa (exponente de dos cifras):
# con la cadena cruda, FLOOR_REL="1e-3" escribia "fig6_suelo_1e-3.json" mientras el
# fichero depositado es "fig6_suelo_1e-03.json", asi que correr el script como lo
# documenta el README dejaba una copia al lado en vez de regenerar la buena.
SALIDA = "/w/results/fig6_suelo_%.0e.json" % FLOOR_REL
json.dump(res, open(SALIDA, "w"), indent=1)
log("=== FINAL (mean +/- std over 5 seeds) ===")
for m in SIZES:
    row=f"D={m:4d}  greedy={det[m]:7.2f}"
    for name in ("uniform","gflownet","oracle"):
        s=res["stats"][m].get(name)
        row += f"  {name}={s['mean']:7.2f}+/-{s['std']:5.2f}(n{s['n']})" if s else f"  {name}=  --"
    log(row)
log("WROTE %s" % SALIDA)