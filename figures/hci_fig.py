# -*- coding: utf-8 -*-
"""Regenerate ONLY pf_hci.pdf. All four bars are READ from the deposit -- the classical
two from results/hci_baseline.json (0.56 mHa on H2O, 9.13 on N2) and the generative two
from results/gfn_ruidoso_fig7.json -- instead of being reconciled by hand.

This figure used to hardcode 9.90 for N2, and the deposit had recorded three different
values for that same quantity (10.28, 9.90 and 9.81), none of them reproducible. The
declared generator gives 9.13, which is what the paper now typesets as 9.1.

Layout identical to fix_figs.py (legend outside, no overlaps)."""
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#8256B4", sky="#56B4E9", grey="#8A8A8A", ink="#222222")
mpl.rcParams.update({"savefig.dpi":300,"font.size":12.5,"font.family":"serif","mathtext.fontset":"cm",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.22,
    "grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False})

# Las barras clasicas se LEEN del deposito. Antes estaban escritas a mano como
# "N2 HCI 10.28 -> 9.90 (verified)", y el generador declarado (calculations/
# hci_baseline.py) da 9.13. El deposito llego a registrar tres valores distintos para
# la misma cantidad -- 10.28, 9.90 y 9.81 -- ninguno reproducible.
import json, os
_h = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, "results", "hci_baseline.json")))
mols=["H$_2$O","N$_2$"]
hci=[_h["h2o"]["hci_mHa"], _h["n2"]["hci_mHa"]]
gfnv=[_h["h2o"]["gfn_mHa"], _h["n2"]["gfn_mHa"]]
fig,ax=plt.subplots(figsize=(6.8,5.2)); x=np.arange(len(mols)); w=0.32
ax.bar(x-w/2,hci,w,color=OI["vermilion"],label="heat-bath CI (classical, noise-free)",edgecolor="white")
ax.bar(x+w/2,gfnv,w,color=OI["purple"],label="GFlowNet-SQD (quantum-sampled, noisy)",edgecolor="white")
ax.axhline(1.6,color=OI["ink"],ls=":",lw=1.3)
ax.text(1.46,1.9,"chemical accuracy (1.6 mHa)",fontsize=9.5,ha="right",va="bottom",color=OI["ink"])
for xx,v in zip(x-w/2,hci): ax.text(xx,v+0.5,f"{v:.1f}",ha="center",fontsize=11,color=OI["vermilion"],fontweight="bold")
for xx,v in zip(x+w/2,gfnv): ax.text(xx,v+0.5,f"{v:.1f}",ha="center",fontsize=11,color=OI["purple"],fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(mols,fontsize=13); ax.set_ylabel("energy error vs exact FCI (mHa)")
ax.set_ylim(0,30); ax.set_title("Decisive test at FCI-verifiable scale ($D=120$)",fontsize=12.5,pad=8)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.11),ncol=1,fontsize=10)
fig.tight_layout(); fig.savefig("/w/pf_hci.pdf",bbox_inches="tight"); fig.savefig("/w/pf_hci.png",bbox_inches="tight",dpi=140)
print("pf_hci regenerated from results/hci_baseline.json: N2 HCI = %.2f mHa" % hci[1])
