# -*- coding: utf-8 -*-
"""Regenerate ONLY pf_crossover.pdf with the VERIFIED 5-seed error bars.
Numbers-audit recompute (5 seeds) -> std = [6.1, 2.4, 2.5, 2.5], not the notebook
placeholder [3.5, 4.0, 4.2, 4.3]. Real bars: larger at zero noise (the -11.7 gap is
less significant there), smaller as noise grows (the positive crossover is the robust part)."""
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#8256B4", sky="#56B4E9", yellow="#F0E442", grey="#8A8A8A", ink="#222222")
mpl.rcParams.update({"savefig.dpi":300,"font.size":12.5,"font.family":"serif","mathtext.fontset":"cm",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.22,
    "grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False})

prod_s=[0,1,2,3]; prod_g=[-11.74,0.30,9.71,9.84]; prod_e=[6.1,2.4,2.5,2.5]   # verified 5-seed s.d.
fig,ax=plt.subplots(figsize=(7.6,5.0))
ax.axhspan(-30,0,color=OI["vermilion"],alpha=0.06); ax.axhspan(0,30,color=OI["green"],alpha=0.06)
ax.axhline(0,color=OI["ink"],lw=1.1)
ax.errorbar(prod_s,prod_g,yerr=prod_e,fmt="s-",color=OI["blue"],lw=2.3,ms=9,capsize=4,
            label="5-seed mean $\\pm$ s.d. (1000 iters)")
# region labels placed OUTSIDE the line's y-range [-11.7, +9.8] so they can never touch the curve/error bars
ax.text(0.12,13.4,"GFlowNet better",color=OI["green"],fontsize=12,fontweight="bold")
ax.text(0.12,-14.6,"classical control better",color=OI["vermilion"],fontsize=12,fontweight="bold")
ax.set_ylim(-20,16); ax.set_xlim(-0.25,3.25); ax.set_xticks([0,1,2,3])
ax.set_xlabel(r"noise scale ($\times$ FakeTorino/Heron)")
ax.set_ylabel(r"gap  $E_{\mathrm{ibm+cheap}}-E_{\mathrm{gfn}}$  (mHa)")
ax.set_title("The noise crossover: generative advantage is\nregime-dependent, emerging only under noise",fontsize=12.5)
ax.legend(fontsize=10.3,loc="lower right"); fig.tight_layout()
fig.savefig("/w/pf_crossover.pdf",bbox_inches="tight"); fig.savefig("/w/pf_crossover.png",bbox_inches="tight",dpi=140)
print("pf_crossover regenerated with verified 5-seed s.d. [6.1,2.4,2.5,2.5]")
