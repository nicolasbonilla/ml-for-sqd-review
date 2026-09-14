# -*- coding: utf-8 -*-
"""Compose the journal-grade Figure 2: MO energy-level diagrams (with electron
occupation) + ray-traced PyMOL orbital/structure panels.

ORBITAL ASSIGNMENT (fixed 2026-09-11). Earlier versions of this file called MO#6
"3sigma_g (HOMO)" at -10.9 eV. That is wrong: for N2 at R=2.0 A / cc-pVDZ the RHF
spectrum is MO#4 = A1g (the genuine, NON-degenerate 3sigma_g) at -12.985 eV, and
MO#5/MO#6 = E1uy/E1ux (the two-fold-degenerate 1pi_u HOMO pair) at -10.899 eV. So
-10.9 eV is the 1pi_u HOMO and -13.0 eV is 3sigma_g -- which is exactly what the
paper caption prints. Panel (d) therefore loads o_n2_sg_true.png (rendered by
render_sg.py from the cube of gen_sg_true.py, MO#4) and NOT the legacy o_n2_sg.png,
which is a cube of MO#6, i.e. a 1pi_u component wearing a sigma label.

Prerequisites: gen_cubes.py (energies + legacy cubes + .xyz), gen_sg_true.py (the
MO#4 cube), then render_orb.py and render_sg.py for the PNGs.
"""
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, Rectangle
HA2EV = 27.211386
OI = dict(occ="#2E4A8B", virt="#B0392B", band="#E9F0FA", bandv="#FBECE9",
          arrow="#1A1A1A", ink="#222", pos="#E69F00", neg="#0072B2", grey="#8A8A8A")
mpl.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":12,
                     "savefig.dpi":300})

def img(ax, path, title=None, sub=None):
    ax.imshow(mpimg.imread(path)); ax.axis("off")
    if title: ax.set_title(title, fontsize=12.5, pad=1)
    if sub:  ax.text(0.5,-0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=OI["grey"])

def mo_diagram(ax, moe, nocc, ncore, ncas, keymarks, title):
    """Energy-level ladder (eV) for the active space, with electron occupation."""
    e = moe*HA2EV
    lo, hi = ncore, ncore+ncas
    act = e[lo:hi]
    ymin, ymax = act.min()-2.0, act.max()+2.0
    # active-space background band split at Fermi (HOMO/LUMO midpoint)
    ef = 0.5*(e[nocc-1]+e[nocc])
    ax.add_patch(Rectangle((0,act.min()-0.6),1,(ef-(act.min()-0.6)),color=OI["band"],zorder=0))
    ax.add_patch(Rectangle((0,ef),1,(act.max()+0.6-ef),color=OI["bandv"],zorder=0))
    ax.axhline(ef, color=OI["grey"], ls="--", lw=1, zorder=1)
    ax.text(0.07, ef, "$E_F$", va="center", ha="left", fontsize=10, color="#555", zorder=6,
            bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.2))
    # group near-degenerate levels to place them side by side
    groups=[]
    for i in range(lo,hi):
        placed=False
        for g in groups:
            if abs(e[i]-e[g[0]])<0.25: g.append(i); placed=True; break
        if not placed: groups.append([i])
    for g in groups:
        n=len(g); w=0.30
        xs=np.linspace(0.5-(n-1)*0.16, 0.5+(n-1)*0.16, n)
        for x,i in zip(xs,g):
            occ = i < nocc
            col = OI["occ"] if occ else OI["virt"]
            ax.plot([x-w/2,x+w/2],[e[i],e[i]], color=col, lw=3, solid_capstyle="round", zorder=3)
            if occ:  # two electrons (up/down arrows)
                for dx,dy in [(-0.055,1),(0.055,-1)]:
                    ax.annotate("", xy=(x+dx,e[i]+0.42*dy), xytext=(x+dx,e[i]-0.42*dy),
                        arrowprops=dict(arrowstyle="-|>",color=OI["arrow"],lw=1.3), zorder=4)
    # key-orbital labels with anti-overlap vertical spreading + connector lines
    items = sorted(keymarks.items(), key=lambda kv: e[kv[0]])
    span = ymax - ymin; gap = 0.11*span
    ys = []
    for k, (i, lab) in enumerate(items):
        yt = e[i]
        if ys and yt < ys[-1] + gap:
            yt = ys[-1] + gap
        ys.append(yt)
    for (i, lab), yt in zip(items, ys):
        ax.annotate(lab, xy=(0.66, e[i]), xytext=(0.90, yt),
                    fontsize=10.5, va="center", ha="left", color=OI["ink"], zorder=6,
                    bbox=dict(fc="white", alpha=0.78, ec="none", pad=0.8),
                    arrowprops=dict(arrowstyle="-", color=OI["grey"], lw=0.8,
                                    connectionstyle="arc3,rad=0.0"))
    ax.set_xlim(0, 1.9); ax.set_ylim(ymin, ymax)
    ax.set_ylabel("orbital energy (eV)", fontsize=11.5)
    ax.set_xticks([]); ax.spines[["top","right","bottom"]].set_visible(False)
    ax.set_title(title, fontsize=12.5)
    ax.plot([],[],color=OI["occ"],lw=3,label="occupied (active)")
    ax.plot([],[],color=OI["virt"],lw=3,label="virtual (active)")
    ax.legend(loc="upper center", fontsize=8.6, frameon=False, bbox_to_anchor=(0.5,-0.01), ncol=2)

n2e=np.load("/w/n2_moe.npy"); h2oe=np.load("/w/h2o_moe.npy")
def ev(a,i): return a[i]*HA2EV

fig=plt.figure(figsize=(14.5,9.2))
gs=fig.add_gridspec(2,5, width_ratios=[1.05,1.25,1,1,1], height_ratios=[1,1],
                    wspace=0.06, hspace=0.22)
# ---- Row 1: N2 ----
img(fig.add_subplot(gs[0,0]), "/w/o_n2_struct.png", r"(a) N$_2$,  $R=2.0\,$Å", "CAS(10e, 12o)")
# keys are MO indices: 4 = 3sigma_g (A1g), 5/6 = the degenerate 1pi_u HOMO pair
# (E1uy/E1ux -- label once, on 6, since both sit at the same energy), 7 = 1pi_g* LUMO.
mo_diagram(fig.add_subplot(gs[0,1]), n2e, 7, 2, 12,
           {4:r"$3\sigma_g$", 6:r"$1\pi_u$ (HOMO)", 7:r"$1\pi_g^{*}$ (LUMO)", 9:r"$3\sigma_u^{*}$"},
           r"(b) N$_2$ MO levels")
img(fig.add_subplot(gs[0,2]), "/w/o_n2_pi.png",      r"(c) $1\pi_u$ HOMO", f"{ev(n2e,5):+.1f} eV")
img(fig.add_subplot(gs[0,3]), "/w/o_n2_sg_true.png", r"(d) $3\sigma_g$",   f"{ev(n2e,4):+.1f} eV")
img(fig.add_subplot(gs[0,4]), "/w/o_n2_pistar.png", r"(e) $1\pi_g^{*}$ LUMO", f"{ev(n2e,7):+.1f} eV")
# ---- Row 2: H2O ----
img(fig.add_subplot(gs[1,0]), "/w/o_h2o_struct.png", r"(f) H$_2$O", "CAS(8e, 12o)")
mo_diagram(fig.add_subplot(gs[1,1]), h2oe, 5, 1, 12,
           {3:r"$3a_1$", 4:r"$1b_1$ lone pair (HOMO)", 5:r"$4a_1^{*}$ (LUMO)"},
           r"(g) H$_2$O MO levels")
img(fig.add_subplot(gs[1,2]), "/w/o_h2o_homo.png", r"(h) $1b_1$ HOMO", f"{ev(h2oe,4):+.1f} eV")
img(fig.add_subplot(gs[1,3]), "/w/o_h2o_lumo.png", r"(i) $4a_1^{*}$ LUMO", f"{ev(h2oe,5):+.1f} eV")
# phase legend panel
axl=fig.add_subplot(gs[1,4]); axl.axis("off")
axl.add_patch(Rectangle((0.12,0.62),0.16,0.10,color=OI["pos"]))
axl.add_patch(Rectangle((0.12,0.44),0.16,0.10,color=OI["neg"]))
axl.text(0.32,0.67,"orbital phase $+$",fontsize=11,va="center")
axl.text(0.32,0.49,"orbital phase $-$",fontsize=11,va="center")
axl.text(0.12,0.30,r"isosurface $|\psi|=0.028\,a_0^{-3/2}$",fontsize=10,color=OI["grey"])
axl.text(0.12,0.20,"ray-traced (PyMOL)",fontsize=10,color=OI["grey"],style="italic")

fig.suptitle("The molecular systems, their active-space orbital-energy structure, and the frontier orbitals",
             fontsize=15, y=0.975)
fig.savefig("/w/pf_system.pdf", bbox_inches="tight")
fig.savefig("/w/pf_system.png", bbox_inches="tight", dpi=130)
print("Figure 2 composed")
