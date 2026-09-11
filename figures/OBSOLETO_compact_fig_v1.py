"""OBSOLETO -- NO REPRODUCE LA FIGURA 6.

Este script usa `np.maximum(w_cheap, 1e-12)` como suelo de la recompensa.
La Figura 6 del paper se hizo con `FLOOR = 1e-3 * w_cheap.max()`, NUEVE
ORDENES DE MAGNITUD mas alto. Con el suelo de aqui salen ~19.2 mHa donde la
figura reporta 192, y la conclusion se invierte.

    ---> use `fig6_5seed.py`, que es el generador real. <---

Se conserva solo como registro de la version v1. Documentado en
gauge_study/README.md ("La trampa que costo mas cara").
"""
# -*- coding: utf-8 -*-
"""Recompute ONLY the compactness figure with all five series clearly visible."""
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1
import torch, torch.nn as nn
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#8256B4", sky="#56B4E9", grey="#8A8A8A", ink="#222222")
mpl.rcParams.update({"savefig.dpi":300,"font.size":12.5,"font.family":"serif","mathtext.fontset":"cm",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.22,
    "grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False})

NCAS=12; NELECAS=(5,5); na,nb=NELECAS
mol=gto.M(atom="N 0 0 0; N 0 0 2.0",basis="cc-pvdz",verbose=0); mf=scf.RHF(mol).run()
cas=mcscf.CASCI(mf,NCAS,NELECAS); h1,ecore=cas.get_h1cas(); h2=ao2mo.restore(1,cas.get_h2cas(),NCAS)
strs_a=cistring.make_strings(range(NCAS),na); dim_a=len(strs_a)
set_to_idx={frozenset(p for p in range(NCAS) if (int(s)>>p)&1):i for i,s in enumerate(strs_a)}
hf_idx=int(np.where(strs_a==(1<<na)-1)[0][0])
e_fci,civec=pyscf.fci.direct_spin1.FCI().kernel(h1,h2,NCAS,NELECAS,ecore=ecore); civec=civec.reshape(dim_a,dim_a)
w_a=(civec**2).sum(1); w_a/=w_a.sum()
civ_hf=np.zeros((dim_a,dim_a)); civ_hf[hf_idx,hf_idx]=1.0
h2e=direct_spin1.absorb_h1e(h1,h2,NCAS,NELECAS,0.5)
Hc=direct_spin1.contract_2e(h2e,civ_hf,NCAS,NELECAS).reshape(dim_a,dim_a)
hdiag=direct_spin1.make_hdiag(h1,h2,NCAS,NELECAS).reshape(dim_a,dim_a)
den=Hc[hf_idx,hf_idx]-hdiag; den[hf_idx,hf_idx]=1.0; c1=Hc/den; c1[hf_idx,hf_idx]=1.0
w_cheap=(c1**2).sum(1); w_cheap/=w_cheap.sum(); cheap_order=np.argsort(w_cheap)[::-1]
_sci=selected_ci.SelectedCI()
def err(idxs):
    s=np.asarray(sorted(set(int(strs_a[i]) for i in idxs)),dtype=np.int64)
    out=selected_ci.kernel_fixed_space(_sci,h1,h2,NCAS,NELECAS,(s,s),ecore=ecore)
    return (float(out[0] if isinstance(out,(tuple,list)) else out)-e_fci)*1000
torch.manual_seed(0); rng=np.random.default_rng(0)
class P(nn.Module):
    def __init__(s,n):
        super().__init__(); s.net=nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,n)); s.logZ=nn.Parameter(torch.zeros(1))
    def forward(s,x): return s.net(x)
def sb(net,B):
    st=torch.zeros(B,NCAS); lpf=torch.zeros(B)
    for _ in range(na):
        lp=torch.log_softmax(net(st).masked_fill(st.bool(),-1e9),1); a=torch.distributions.Categorical(logits=lp).sample()
        lpf+=lp.gather(1,a[:,None]).squeeze(1); st=st.scatter(1,a[:,None],1.0)
    return np.array([set_to_idx[frozenset(np.where(st[b].numpy()>0)[0].tolist())] for b in range(B)]),lpf
def traindraw(reward,iters=1500,ndraw=60000):
    net=P(NCAS); opt=torch.optim.Adam([{"params":net.net.parameters(),"lr":1e-3},{"params":[net.logZ],"lr":1e-1}])
    Rt=torch.tensor(reward/reward.sum(),dtype=torch.float32)
    for it in range(iters):
        idxs,lpf=sb(net,256); loss=((net.logZ+lpf-torch.log(Rt[idxs]+1e-12))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    o=[]
    while len(o)<ndraw:
        idxs,_=sb(net,512); o+=idxs.tolist()
    return o
sizes=[30,60,120,240]
def compact(draws):
    seen=set(); out={}; tgt=sorted(sizes); j=0
    for d in draws:
        seen.add(int(d))
        if j<len(tgt) and len(seen)>=tgt[j]: out[tgt[j]]=err(seen); j+=1
    return out
w_temp=np.maximum(w_cheap,1e-12)**0.5; w_temp/=w_temp.sum()   # tempered (beta=0.5) -> diverse
gfn_draws=traindraw(w_temp,iters=1500,ndraw=300000)
unif=compact(rng.integers(0,dim_a,20000))
ciid=compact(rng.choice(dim_a,300000,p=w_cheap))              # naive i.i.d. on the peaked cheap reward -> saturates
gfn=compact(gfn_draws)
orac=compact(rng.choice(dim_a,300000,p=w_a))
hcig={m:err(cheap_order[:m]) for m in sizes}
print("uniq gfn:",len(set(gfn_draws)))
for nm,d in [("unif",unif),("ciid",ciid),("hci",hcig),("gfn",gfn),("orac",orac)]:
    print(nm,{k:round(v,2) for k,v in d.items()})

fig,ax=plt.subplots(figsize=(7.6,5.1))
# oracle as a light dashed guide-line UNDER everything
xo=[m for m in sizes if m in orac]; yo=[orac[m] for m in xo]
ax.plot(xo,yo,ls="--",color=OI["green"],lw=1.8,zorder=1,alpha=0.9)
S=[("uniform (blind)",unif,OI["grey"],"o",2),
   (r"greedy top-$K$ (static cheap reward)",hcig,OI["vermilion"],"D",4),
   ("GFlowNet on cheap reward (ours)",gfn,OI["purple"],"^",6),
   (r"oracle $\propto |c|^2$ (unreachable)",orac,OI["green"],"*",5)]
for name,d,c,mk,zz in S:
    xs=[m for m in sizes if m in d]; ys=[d[m] for m in xs]
    ax.plot(xs,ys,marker=mk,ls="none" if mk=="*" else "-",color=c,lw=2,
            ms=16 if mk=="*" else 9.5,label=name,zorder=zz,
            markeredgecolor="white",markeredgewidth=1.1)
ax.axhline(1.6,color=OI["ink"],ls=":",lw=1.3,zorder=0); ax.text(246,1.72,"chemical accuracy",fontsize=10,ha="right")
ax.set_yscale("log"); ax.set_xlabel("subspace dimension (number of strings)")
ax.set_ylabel("energy error vs exact FCI (mHa)"); ax.set_xlim(15,255)
ax.set_title("A cheap, FCI-free GFlowNet tracks the exact oracle\n(naive i.i.d. sampling saturates below $D=30$; the decisive iterative-HCI test is shown separately)",fontsize=11.8)
ax.legend(fontsize=9.6,loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=2)  # OUTSIDE, below axes
fig.tight_layout()
fig.savefig("/w/pf_compact.pdf",bbox_inches="tight"); fig.savefig("/w/pf_compact.png",bbox_inches="tight",dpi=140)
print("compact fig fixed")
