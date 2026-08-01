# -*- coding: utf-8 -*-
"""Workflow figure v5 — self-contained image panels (zero text/arrow overlap),
now with a heavy-hex QPU lattice and a real eigenvalue spectrum for diagonalization."""
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from scipy.sparse.linalg import LinearOperator, eigsh
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1

OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#8256B4", grey="#8A8A8A", ink="#1c2733", qband="#eef5fc", cband="#f6f0fb",
          qdot="#0072B2", qact="#E69F00", cpl="#9db8d6")
mpl.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","savefig.dpi":150})
PF=(3.5,2.55)

# ---------------- real data ----------------
NCAS=12; NELECAS=(5,5); na,nb=NELECAS
mol=gto.M(atom="N 0 0 0; N 0 0 2.0",basis="cc-pvdz",verbose=0); mf=scf.RHF(mol).run()
cas=mcscf.CASCI(mf,NCAS,NELECAS); h1,ecore=cas.get_h1cas(); h2=ao2mo.restore(1,cas.get_h2cas(),NCAS)
strs_a=cistring.make_strings(range(NCAS),na); dim_a=len(strs_a)
str_to_idx={int(s):i for i,s in enumerate(strs_a)}
e_fci,civec=pyscf.fci.direct_spin1.FCI().kernel(h1,h2,NCAS,NELECAS,ecore=ecore); civec=civec.reshape(dim_a,dim_a)
w_a=(civec**2).sum(1); w_a/=w_a.sum(); order=np.argsort(w_a)[::-1]
h2e=direct_spin1.absorb_h1e(h1,h2,NCAS,NELECAS,0.5)
rng=np.random.default_rng(1); shots=4000
ideal=rng.choice(dim_a,size=shots,p=w_a)
bits=np.array([[(int(strs_a[i])>>p)&1 for p in range(NCAS)] for i in ideal],dtype=np.int8)
dep=rng.random(shots)<0.05; bits[dep]=(rng.random((dep.sum(),NCAS))<0.5).astype(np.int8)
o,z=bits==1,bits==0; bits[o&(rng.random(bits.shape)<0.023)]=0; bits[z&(rng.random(bits.shape)<0.02)]=1
valid=np.array([b.sum()==na for b in bits]); frac_valid=valid.mean(); occ=bits[valid].mean(0)
cv={}
for b in bits[valid]:
    s=int(sum(int(v)<<p for p,v in enumerate(b))); cv[str_to_idx.get(s,-1)]=cv.get(str_to_idx.get(s,-1),0)+1
cv.pop(-1,None); topf=sorted(cv.items(),key=lambda kv:-kv[1])[:12]
occ_mat=np.array([[(int(strs_a[i])>>p)&1 for p in range(NCAS)] for i in order[:28]])
# eigenvalue spectrum of the projected Hamiltonian in a subspace
idx=np.asarray(sorted(set(int(i) for i in order[:45]))); nS=len(idx)
def mv(x):
    full=np.zeros((dim_a,dim_a)); full[np.ix_(idx,idx)]=x.reshape(nS,nS)
    return direct_spin1.contract_2e(h2e,full,NCAS,NELECAS).reshape(dim_a,dim_a)[np.ix_(idx,idx)].ravel()
evals,_=eigsh(LinearOperator((nS*nS,nS*nS),matvec=mv),k=10,which="SA",maxiter=4000,tol=1e-7)
evals=np.sort(evals)+ecore                     # total energies (Ha)
erel=(evals-evals[0])*27.2114                   # eV above ground

# ---------------- render each panel as a tight PNG ----------------
def finish(f,fn): f.tight_layout(pad=0.5); f.savefig(fn,dpi=150,facecolor="white"); plt.close(f)

# heavy-hex QPU lattice (stylized IBM Heron/Torino topology)
f,a=plt.subplots(figsize=PF); a.set_xlim(-0.5,6.5); a.set_ylim(-0.5,4.5); a.axis("off"); a.set_aspect("equal")
main_rows=[0,2,4]; nodes=[]; edges=[]
for ri,y in enumerate(main_rows):
    for x in range(7): nodes.append((x,y))
    for x in range(6): edges.append(((x,y),(x+1,y)))
for yi,y in enumerate([1,3]):                    # connector rows (heavy-hex: alternating columns)
    xs=range(0,7,2) if yi==0 else range(1,7,2)
    for x in xs:
        nodes.append((x,y)); edges.append(((x,y-1),(x,y))); edges.append(((x,y),(x,y+1)))
for (p,q) in edges: a.plot([p[0],q[0]],[p[1],q[1]],color=OI["cpl"],lw=2.4,zorder=1,solid_capstyle="round")
active=set([(2,2),(3,2),(4,2),(2,1),(4,1),(2,3),(3,4)])   # a highlighted LUCJ patch
for (x,y) in nodes:
    on=(x,y) in active
    a.add_patch(Circle((x,y),0.20,fc=OI["qact"] if on else OI["qdot"],ec="white",lw=1.4,zorder=3))
a.set_title("heavy-hex qubit lattice · LUCJ patch (orange)",fontsize=9.5,color=OI["grey"],pad=6)
finish(f,"/w/wp_qpu.png")

# histogram
f,a=plt.subplots(figsize=PF); vals=[c for _,c in topf]
a.barh(np.arange(len(vals))[::-1],vals,color=OI["blue"],height=0.75)
a.set_yticks([]); a.set_xlabel("counts",fontsize=11); a.set_ylabel("sampled configurations\n(ranked by frequency)",fontsize=10.5); a.tick_params(labelsize=10)
for s in ["top","right"]: a.spines[s].set_visible(False)
a.text(0.97,0.10,f"{100*frac_valid:.0f}% valid shots\n(Heron noise)",transform=a.transAxes,ha="right",va="bottom",fontsize=10.5,color=OI["grey"])
finish(f,"/w/wp_hist.png")

# occupation
f,a=plt.subplots(figsize=PF); a.bar(range(NCAS),occ,color=OI["purple"],width=0.82); a.axhline(na/NCAS,color=OI["grey"],ls="--",lw=1.2)
a.set_xlabel("orbital $p$",fontsize=11); a.set_ylabel(r"mean occupation $\langle n_p\rangle$",fontsize=11); a.set_ylim(0,1.05); a.tick_params(labelsize=10)
for s in ["top","right"]: a.spines[s].set_visible(False)
finish(f,"/w/wp_occ.png")

# matrix
f,a=plt.subplots(figsize=PF); a.imshow(occ_mat,cmap="Blues",aspect="auto",interpolation="nearest")
a.set_xlabel("orbital (12)",fontsize=11); a.set_ylabel("selected determinants",fontsize=11); a.tick_params(labelsize=10); a.set_xticks([0,5,11]); a.set_yticks([0,13,27])
finish(f,"/w/wp_matrix.png")

# eigenvalue spectrum
f,a=plt.subplots(figsize=PF)
for k,e in enumerate(erel):
    c=OI["green"] if k==0 else OI["grey"]; lw=3.2 if k==0 else 1.8
    a.plot([0.3,0.7],[e,e],color=c,lw=lw,solid_capstyle="round")
a.annotate("$E_0$ (ground state)",xy=(0.7,erel[0]),xytext=(0.78,erel[0]),fontsize=10.5,va="center",color=OI["green"],fontweight="bold")
a.text(0.5,erel[-1]+0.5,"excited states",ha="center",fontsize=9.5,color=OI["grey"])
a.set_xlim(0,1.7); a.set_ylim(-0.8,erel[-1]+1.5); a.set_xticks([])
a.set_ylabel("energy above $E_0$ (eV)",fontsize=11); a.tick_params(labelsize=10)
for s in ["top","right","bottom"]: a.spines[s].set_visible(False)
finish(f,"/w/wp_spec.png")

# ---------------- compose ----------------
fig=plt.figure(figsize=(14,9.4))
bg=fig.add_axes([0,0,1,1]); bg.set_xlim(0,1); bg.set_ylim(0,1); bg.axis("off")
colx=[0.055,0.375,0.695]; rw=0.25; rowy=[0.61,0.20]; rh=0.25
def place(png,c,r):
    a=fig.add_axes([colx[c],rowy[r],rw,rh]); a.imshow(mpimg.imread(png)); a.axis("off")
# bands
bg.add_patch(FancyBboxPatch((0.028,0.585),0.944,0.375,boxstyle="round,pad=0.006,rounding_size=0.016",fc=OI["qband"],ec=OI["blue"],lw=1.5,zorder=0))
bg.add_patch(FancyBboxPatch((0.028,0.165),0.944,0.375,boxstyle="round,pad=0.006,rounding_size=0.016",fc=OI["cband"],ec=OI["purple"],lw=1.5,zorder=0))
bg.text(0.5,0.935,"QUANTUM PROCESSOR",color=OI["blue"],fontsize=13.5,fontweight="bold",ha="center")
bg.text(0.5,0.515,"CLASSICAL CO-PROCESSOR (HPC)",color=OI["purple"],fontsize=13.5,fontweight="bold",ha="center")
def title(c,r,t): bg.text(colx[c]+rw/2, rowy[r]+rh+0.016, t, ha="center", fontsize=12.5, fontweight="bold", color=OI["ink"])
for t,c,r in [("1 · Molecule + Hamiltonian",0,0),("2 · LUCJ circuit on QPU",1,0),("3 · Sample bitstrings",2,0),
              ("4 · S-CORE recovery",2,1),("5 · Determinant subspace $S$",1,1),("6 · Diagonalize $\\to E_0$",0,1)]:
    title(c,r,t)
place("/w/o_n2_struct.png",0,0); place("/w/wp_qpu.png",1,0); place("/w/wp_hist.png",2,0)
place("/w/wp_occ.png",2,1); place("/w/wp_matrix.png",1,1); place("/w/wp_spec.png",0,1)
def arr(p0,p1,lw=2.8,ms=26):
    bg.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=OI["ink"],zorder=6))
ym1=rowy[0]+rh/2; ym2=rowy[1]+rh/2
arr((colx[0]+rw+0.004,ym1),(colx[1]-0.006,ym1))
arr((colx[1]+rw+0.004,ym1),(colx[2]-0.006,ym1))
# 3->4 : down the RIGHT MARGIN, clear of the centred "4 ·" title
arr((colx[2]+rw+0.013,rowy[0]-0.004),(colx[2]+rw+0.013,rowy[1]+rh+0.004))
bg.text(colx[2]+rw+0.028,(rowy[0]+rowy[1]+rh)/2,"measured\nbitstrings",fontsize=9.5,color=OI["grey"],va="center",ha="left",rotation=90)
arr((colx[2]-0.006,ym2),(colx[1]+rw+0.004,ym2))
arr((colx[1]-0.006,ym2),(colx[0]+rw+0.004,ym2))
# footer BELOW the classical band, with a separator
bg.plot([0.20,0.80],[0.115,0.115],color=OI["grey"],lw=0.8,alpha=0.5)
bg.text(0.5,0.083,"Machine-learning entry points",ha="center",fontsize=11.5,fontweight="bold",color=OI["ink"])
def badge(x,y,num,color):
    bg.add_patch(Circle((x,y),0.014,color=color,zorder=9)); bg.text(x,y,str(num),ha="center",va="center",fontsize=10,color="white",fontweight="bold",zorder=10)
badge(0.145,0.042,1,OI["orange"]); bg.text(0.164,0.042,"design the sampling circuit  (GQE, stage 2)",va="center",ha="left",fontsize=10.5,color=OI["ink"])
badge(0.560,0.042,2,OI["purple"]); bg.text(0.579,0.042,"generate / select configurations — RBM · NQS · GFlowNet  (stages 4–5)",va="center",ha="left",fontsize=10.5,color=OI["ink"])
fig.savefig("/w/pf_workflow.pdf",bbox_inches="tight"); fig.savefig("/w/pf_workflow.png",bbox_inches="tight",dpi=125)
print("workflow v5 done valid%=",round(100*frac_valid,1),"gap E1-E0=",round(erel[1],2),"eV")
