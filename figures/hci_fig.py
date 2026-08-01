# -*- coding: utf-8 -*-
"""Regenerate ONLY pf_hci.pdf with the reconciled N2 HCI value (9.90 mHa, matching the
companion notebook; independent hci_baseline.py recompute confirms 9.81 -> both round to the
text's '~10 mHa'). Layout identical to fix_figs.py (legend outside, no overlaps)."""
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#8256B4", sky="#56B4E9", grey="#8A8A8A", ink="#222222")
mpl.rcParams.update({"savefig.dpi":300,"font.size":12.5,"font.family":"serif","mathtext.fontset":"cm",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.22,
    "grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False})

mols=["H$_2$O","N$_2$"]; hci=[0.56,9.90]; gfnv=[2.1,26.7]      # N2 HCI 10.28 -> 9.90 (verified)
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
print("pf_hci regenerated with N2 HCI = 9.90 mHa")
