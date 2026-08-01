# -*- coding: utf-8 -*-
"""Visual taxonomy of ML-for-SQD methods: generated object x importance signal.
Schematic; chips placed on a grid with generous spacing (no overlaps)."""
import matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
OI = dict(gen="#8256B4", disc="#E69F00", circ="#009E73", ours="#D55E00",
          grid="#D9D9D9", ink="#1c2733", grey="#8A8A8A", star="#D55E00", hl="#FDEEE6")
mpl.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","savefig.dpi":300})

# columns = importance signal ; rows = generated object
COLS=["Cheap physics prior\n(Epstein–Nesbet PT)","Quantum-sample\nfrequency",
      "Hamiltonian coupling\n/ variational","Reward-proportional\n(RL / flow)"]
ROWS=["Sampling circuit","Determinant\ndistribution","Individual\ndeterminants (in loop)","Determinant label\n/ ranking"]
# (col,row): list of (name, family)
CELLS={
 (3,0):[("GQE / Generative-QSCI\n[Kemmoku 2026]","circ")],
 (1,1):[("RBM PIGen-SQD\n[Maitra 2025]","gen"),("QSCI-RBM (DMET)\n[Maitra 2026]","gen")],
 (2,2):[("HI-NQS [2026]","gen"),("NQS-SC [2026]","gen"),("QiankunNet / NNQS-SCI","gen")],
 (0,3):[("MLCI [Coe 2018]","disc")],
 (2,3):[("cSQD [2026]","disc"),("HAAR-SCI [2025]","disc"),("RCI rank [2026]","disc")],
 (3,2):[("GFlowNet — valid-$N$ by\nconstruction  (this review)","ours")],
}
FAM={"gen":OI["gen"],"disc":OI["disc"],"circ":OI["circ"],"ours":OI["ours"]}

nC,nR=len(COLS),len(ROWS)
fig,ax=plt.subplots(figsize=(13,7.6)); ax.set_xlim(0,nC); ax.set_ylim(0,nR); ax.axis("off")
x0=0.0
# grid cells
for c in range(nC):
    for r in range(nR):
        y=nR-1-r
        hl = (c,r)==(3,2)
        ax.add_patch(FancyBboxPatch((c+0.04,y+0.04),0.92,0.92,boxstyle="round,pad=0.0,rounding_size=0.04",
                     fc=OI["hl"] if hl else "white",ec=OI["star"] if hl else OI["grid"],
                     lw=2.2 if hl else 1.1,ls="--" if hl else "-",zorder=1))
# column headers (top, above grid)
for c in range(nC):
    ax.text(c+0.5,nR+0.28,COLS[c],ha="center",va="center",fontsize=11,fontweight="bold",color=OI["ink"])
# row headers (left, outside grid)
for r in range(nR):
    y=nR-1-r
    ax.text(-0.12,y+0.5,ROWS[r],ha="right",va="center",fontsize=11,fontweight="bold",color=OI["ink"])
# title (top strip, clear of everything)
ax.text(1.35,nR+1.35,"A taxonomy of machine-learning methods for determinant\nand subspace selection in SQD",
        ha="center",va="center",fontsize=13.5,fontweight="bold",color=OI["ink"])
# axis arrows
ax.annotate("",xy=(nC+0.02,nR+0.60),xytext=(0,nR+0.60),arrowprops=dict(arrowstyle="-|>",color=OI["grey"],lw=1.6))
ax.text(nC/2,nR+0.74,"importance signal exploited  →",ha="center",fontsize=11,style="italic",color=OI["grey"])
ax.annotate("",xy=(-1.28,0.0),xytext=(-1.28,nR),arrowprops=dict(arrowstyle="-|>",color=OI["grey"],lw=1.6))
ax.text(-1.45,nR/2,"generated object  →",rotation=90,ha="center",va="center",fontsize=11.5,style="italic",color=OI["grey"])

def chip(cx,cy,txt,fam,big=False):
    col=FAM[fam]
    w,h=0.86,0.30 if not big else 0.40
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.006,rounding_size=0.05",
                 fc=col,ec="white",lw=1.2,alpha=0.92 if fam!="ours" else 1.0,zorder=3))
    ax.text(cx,cy,txt,ha="center",va="center",fontsize=8.6,color="white",fontweight="bold",zorder=4)

for (c,r),items in CELLS.items():
    y=nR-1-r; n=len(items)
    ys=[y+0.5] if n==1 else [y+0.5+0.30*(n-1)/2-0.30*k for k in range(n)]
    for (name,fam),cy in zip(items,ys):
        chip(c+0.5,cy,name,fam,big=(fam=="ours"))
# star on the whitespace cell
ax.plot(3+0.5,nR-1-2+0.14,marker="*",ms=20,color=OI["star"],zorder=5,markeredgecolor="white",markeredgewidth=0.8)

# legend (below, outside grid) — families
leg=[Line2D([0],[0],marker="s",ls="none",ms=12,mfc=OI["gen"],mec="white",label="Generative (distribution / NQS)"),
     Line2D([0],[0],marker="s",ls="none",ms=12,mfc=OI["disc"],mec="white",label="Discriminative (classifier / ranking)"),
     Line2D([0],[0],marker="s",ls="none",ms=12,mfc=OI["circ"],mec="white",label="Circuit design"),
     Line2D([0],[0],marker="s",ls="none",ms=12,mfc=OI["ours"],mec="white",label="Reward-proportional flow (whitespace)")]
ax.legend(handles=leg,loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=2,fontsize=10.5,frameon=False)
fig.savefig("/w/pf_taxonomy.pdf",bbox_inches="tight"); fig.savefig("/w/pf_taxonomy.png",bbox_inches="tight",dpi=140)
print("taxonomy figure done")
