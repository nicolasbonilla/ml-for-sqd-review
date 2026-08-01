# -*- coding: utf-8 -*-
"""Recompute the coupon-collector figure with non-overlapping annotations."""
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#8256B4", ink="#222222")
mpl.rcParams.update({"savefig.dpi":300,"font.size":12.5,"font.family":"serif","mathtext.fontset":"cm",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.22,
    "grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False})
NCAS=12; NELECAS=(5,5); na,nb=NELECAS
mol=gto.M(atom="N 0 0 0; N 0 0 2.0",basis="cc-pvdz",verbose=0); mf=scf.RHF(mol).run()
cas=mcscf.CASCI(mf,NCAS,NELECAS); h1,ecore=cas.get_h1cas(); h2=ao2mo.restore(1,cas.get_h2cas(),NCAS)
strs_a=cistring.make_strings(range(NCAS),na); dim_a=len(strs_a)
e_fci,civec=pyscf.fci.direct_spin1.FCI().kernel(h1,h2,NCAS,NELECAS,ecore=ecore); civec=civec.reshape(dim_a,dim_a)
w_a=(civec**2).sum(1); w_a/=w_a.sum(); order=np.argsort(w_a)[::-1]; cum=np.cumsum(w_a[order])
frac_k={f:int(np.searchsorted(cum,f))+1 for f in (0.90,0.99,0.999)}
def panel(ax,lab): ax.text(-0.13,1.06,lab,transform=ax.transAxes,fontsize=15,fontweight="bold",va="top")
r=np.arange(1,dim_a+1)
fig,ax=plt.subplots(1,2,figsize=(11.4,4.5))
ax[0].fill_between(r[frac_k[0.90]-1:],w_a[order][frac_k[0.90]-1:],1e-12,color=OI["orange"],alpha=0.18,label="the correlation-energy tail")
ax[0].semilogy(r,w_a[order],color=OI["blue"],lw=2)
ax[0].axvline(frac_k[0.90],color=OI["orange"],ls="--",lw=1.6)
ax[0].set_ylim(1e-9,1); ax[0].set_xlim(0,dim_a)
ax[0].set_xlabel("single-spin string, ranked by weight"); ax[0].set_ylabel(r"marginal weight $w_\alpha(i)$")
ax[0].set_title("Heavy-tailed determinant weights",fontsize=13)
ax[0].annotate(f"90% weight\nin {frac_k[0.90]} strings",xy=(frac_k[0.90],w_a[order][frac_k[0.90]-1]),
    xytext=(95,3e-3),fontsize=11,color=OI["vermilion"],arrowprops=dict(arrowstyle="->",color=OI["vermilion"],lw=1.4))
ax[0].legend(loc="upper right",fontsize=10.5); panel(ax[0],"a")
# right panel: cumulative, annotations parked in the empty lower-right
ax[1].plot(r,cum,color=OI["green"],lw=2.4)
labels=[(0.90,OI["orange"],0.50),(0.99,OI["vermilion"],0.35),(0.999,OI["purple"],0.20)]
for f,c,ytxt in labels:
    k=frac_k[f]; ax[1].plot(k,f,"o",color=c,ms=9,zorder=5)
    ax[1].annotate(f"{f*100:.1f}% of weight:  {k} strings  ({100*k/dim_a:.1f}%)",xy=(k,f),
        xytext=(300,ytxt),fontsize=10.8,color=c,va="center",
        arrowprops=dict(arrowstyle="->",color=c,lw=1.1,alpha=0.8,connectionstyle="arc3,rad=-0.2"))
ax[1].set_xlabel("number of strings included"); ax[1].set_ylabel("cumulative ground-state weight")
ax[1].set_ylim(0,1.03); ax[1].set_xlim(0,dim_a)
ax[1].set_title("The coupon-collector signature",fontsize=13); panel(ax[1],"b")
fig.tight_layout(); fig.savefig("/w/pf_coupon.pdf",bbox_inches="tight"); fig.savefig("/w/pf_coupon.png",bbox_inches="tight",dpi=140)
print("coupon fig fixed:",frac_k)
