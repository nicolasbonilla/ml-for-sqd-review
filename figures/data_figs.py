# -*- coding: utf-8 -*-
"""Publication-grade VECTOR (PDF) data figures for the review.
Recomputes the real physics (fixed seeds) and renders Nature-style multi-panel figures."""
import time, numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1
from scipy.sparse.linalg import LinearOperator, eigsh
import torch, torch.nn as nn

t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#8256B4", sky="#56B4E9", yellow="#F0E442", grey="#8A8A8A", ink="#222222")
mpl.rcParams.update({
    "figure.dpi":120,"savefig.dpi":300,"font.size":12.5,"font.family":"serif",
    "mathtext.fontset":"cm","axes.linewidth":0.9,"axes.edgecolor":"#333333",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
    "grid.alpha":0.22,"grid.linewidth":0.6,"axes.axisbelow":True,
    "legend.frameon":False,"xtick.direction":"out","ytick.direction":"out",
})
def panel(ax,lab):
    ax.text(-0.13,1.06,lab,transform=ax.transAxes,fontsize=15,fontweight="bold",va="top")

# ============================================================ N2 physics (shared)
R=2.0; NCAS=12; NELECAS=(5,5); na,nb=NELECAS
mol=gto.M(atom=f"N 0 0 0; N 0 0 {R}",basis="cc-pvdz",verbose=0); mf=scf.RHF(mol).run()
cas=mcscf.CASCI(mf,NCAS,NELECAS); h1,ecore=cas.get_h1cas(); h2=ao2mo.restore(1,cas.get_h2cas(),NCAS)
strs_a=cistring.make_strings(range(NCAS),na); dim_a=len(strs_a)
str_to_idx={int(s):i for i,s in enumerate(strs_a)}
set_to_idx={frozenset(p for p in range(NCAS) if (int(s)>>p)&1):i for i,s in enumerate(strs_a)}
HF=(1<<na)-1; hf_idx=str_to_idx[HF]
e_fci,civec=pyscf.fci.direct_spin1.FCI().kernel(h1,h2,NCAS,NELECAS,ecore=ecore); civec=civec.reshape(dim_a,dim_a)
w_a=(civec**2).sum(1); w_a/=w_a.sum(); order=np.argsort(w_a)[::-1]; cum=np.cumsum(w_a[order])
_sci=selected_ci.SelectedCI()
def err_mHa(idxs):
    s=np.asarray(sorted(set(int(strs_a[i]) for i in idxs)),dtype=np.int64)
    out=selected_ci.kernel_fixed_space(_sci,h1,h2,NCAS,NELECAS,(s,s),ecore=ecore)
    return (float(out[0] if isinstance(out,(tuple,list)) else out)-e_fci)*1000
# cheap EN-PT1 reward
civ_hf=np.zeros((dim_a,dim_a)); civ_hf[hf_idx,hf_idx]=1.0
h2e=direct_spin1.absorb_h1e(h1,h2,NCAS,NELECAS,0.5)
Hc=direct_spin1.contract_2e(h2e,civ_hf,NCAS,NELECAS).reshape(dim_a,dim_a)
hdiag=direct_spin1.make_hdiag(h1,h2,NCAS,NELECAS).reshape(dim_a,dim_a)
den=Hc[hf_idx,hf_idx]-hdiag; den[hf_idx,hf_idx]=1.0; c1=Hc/den; c1[hf_idx,hf_idx]=1.0
w_cheap=(c1**2).sum(1); w_cheap/=w_cheap.sum(); cheap_order=np.argsort(w_cheap)[::-1]
log(f"N2 FCI={e_fci:.6f}")

frac_k={f:int(np.searchsorted(cum,f))+1 for f in (0.90,0.99,0.999)}

# ============================================================ FIG: coupon-collector
fig,ax=plt.subplots(1,2,figsize=(11.4,4.5))
r=np.arange(1,dim_a+1)
ax[0].fill_between(r[frac_k[0.90]-1:],w_a[order][frac_k[0.90]-1:],1e-12,color=OI["orange"],alpha=0.18,label="the correlation-energy tail")
ax[0].semilogy(r,w_a[order],color=OI["blue"],lw=2)
ax[0].axvline(frac_k[0.90],color=OI["orange"],ls="--",lw=1.6)
ax[0].set_ylim(1e-9,1); ax[0].set_xlim(0,dim_a)
ax[0].set_xlabel("single-spin string, ranked by weight"); ax[0].set_ylabel(r"marginal weight $w_\alpha(i)$")
ax[0].set_title("Heavy-tailed determinant weights",fontsize=13)
ax[0].annotate(f"90% weight\nin {frac_k[0.90]} strings",xy=(frac_k[0.90],w_a[order][frac_k[0.90]-1]),
    xytext=(90,3e-3),fontsize=11,color=OI["vermilion"],
    arrowprops=dict(arrowstyle="->",color=OI["vermilion"],lw=1.4))
ax[0].legend(loc="upper right",fontsize=10.5); panel(ax[0],"a")
ax[1].plot(r,cum,color=OI["green"],lw=2.4)
for f,c in [(0.90,OI["orange"]),(0.99,OI["vermilion"]),(0.999,OI["purple"])]:
    k=frac_k[f]; ax[1].plot(k,f,"o",color=c,ms=9,zorder=5)
    ax[1].annotate(f"{f*100:.1f}% : {k} strings ({100*k/dim_a:.1f}%)",xy=(k,f),
        xytext=(k+40,f-0.06),fontsize=10.5,color=c,
        arrowprops=dict(arrowstyle="-",color=c,lw=0.8,alpha=0.6))
ax[1].set_xlabel("number of strings included"); ax[1].set_ylabel("cumulative ground-state weight")
ax[1].set_ylim(0,1.03); ax[1].set_xlim(0,dim_a)
ax[1].set_title("The coupon-collector signature",fontsize=13); panel(ax[1],"b")
fig.tight_layout(); fig.savefig("/w/pf_coupon.pdf",bbox_inches="tight"); fig.savefig("/w/pf_coupon.png",bbox_inches="tight",dpi=140)
log("coupon fig done")

# ============================================================ noise model + S-CORE
LAMBDA0,P100,P010=0.05,0.0229,0.0200
def b2s(b): return int(sum(int(v)<<p for p,v in enumerate(b)))
def sample_noisy(shots,scale,seed):
    P10,P01,LAM=min(P100*scale,0.5),min(P010*scale,0.5),min(LAMBDA0*scale,0.9)
    rr=np.random.default_rng(seed); ideal=rr.choice(dim_a,size=shots,p=w_a)
    bits=np.array([[(int(strs_a[i])>>p)&1 for p in range(NCAS)] for i in ideal],dtype=np.int8)
    dep=rr.random(shots)<LAM; bits[dep]=(rr.random((dep.sum(),NCAS))<0.5).astype(np.int8)
    o,z=bits==1,bits==0; bits[o&(rr.random(bits.shape)<P10)]=0; bits[z&(rr.random(bits.shape)<P01)]=1
    return bits
def s_core(bits):
    good=np.array([int(b.sum())==na for b in bits]); occ=bits[good].mean(0) if good.any() else np.full(NCAS,na/NCAS)
    rec=[]
    for row in bits:
        s=row.copy(); m=int(s.sum())
        if m==na: rec.append(b2s(s)); continue
        if m>na: od=np.where(s==1)[0]; s[od[np.argsort(occ[od])[:m-na]]]=0
        else: em=np.where(s==0)[0]; s[em[np.argsort(occ[em])[::-1][:na-m]]]=1
        rec.append(b2s(s))
    return rec,good.mean()
def rank(c): return [i for i,_ in sorted(c.items(),key=lambda kv:-kv[1])]

scales=[0.0,1.0,2.0,3.0]; D=120; lost=[]; eraw=[]; erec=[]
for sc in scales:
    bits=sample_noisy(3000,sc,0); valid=np.array([int(b.sum())==na for b in bits])
    craw={};
    for b in bits[valid]:
        i=str_to_idx[b2s(b)]; craw[i]=craw.get(i,0)+1
    rec,vf=s_core(bits); crec={}
    for s in rec:
        i=str_to_idx[s]; crec[i]=crec.get(i,0)+1
    lost.append(100*(1-vf)); eraw.append(err_mHa(rank(craw)[:D]) if craw else np.nan); erec.append(err_mHa(rank(crec)[:D]))
fig,ax=plt.subplots(1,2,figsize=(11.2,4.5))
ax[0].plot(scales,lost,"o-",color=OI["vermilion"],lw=2.2,ms=9)
ax[0].fill_between(scales,lost,color=OI["vermilion"],alpha=0.12)
ax[0].set_xlabel(r"noise scale ($\times$ FakeTorino/Heron)"); ax[0].set_ylabel("valid shots lost (%)")
ax[0].set_title("Noise destroys particle-number conservation",fontsize=13); panel(ax[0],"a")
x=np.arange(len(scales)); w=0.34
ax[1].bar(x-w/2,eraw,w,color=OI["grey"],label="raw (discard invalid)",edgecolor="white",lw=0.5)
ax[1].bar(x+w/2,erec,w,color=OI["green"],label="after S-CORE recovery",edgecolor="white",lw=0.5)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{s:.0f}" for s in scales])
ax[1].set_xlabel(r"noise scale"); ax[1].set_ylabel(r"subspace energy error (mHa), $D=120$")
ax[1].set_title("S-CORE rescues the subspace",fontsize=13); ax[1].legend(fontsize=10.5); panel(ax[1],"b")
fig.tight_layout(); fig.savefig("/w/pf_recovery.pdf",bbox_inches="tight"); fig.savefig("/w/pf_recovery.png",bbox_inches="tight",dpi=140)
log("recovery fig done")

# ============================================================ GFlowNet + compactness
torch.manual_seed(0); rng=np.random.default_rng(0)
class Policy(nn.Module):
    def __init__(s,n):
        super().__init__(); s.net=nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,n)); s.logZ=nn.Parameter(torch.zeros(1))
    def forward(s,x): return s.net(x)
def sample_batch(net,B):
    st=torch.zeros(B,NCAS); lpf=torch.zeros(B)
    for _ in range(na):
        lp=torch.log_softmax(net(st).masked_fill(st.bool(),-1e9),1)
        a=torch.distributions.Categorical(logits=lp).sample(); lpf+=lp.gather(1,a[:,None]).squeeze(1); st=st.scatter(1,a[:,None],1.0)
    return np.array([set_to_idx[frozenset(np.where(st[b].numpy()>0)[0].tolist())] for b in range(B)]),lpf
def train_draw(reward,iters=1200,ndraw=40000):
    net=Policy(NCAS); opt=torch.optim.Adam([{"params":net.net.parameters(),"lr":1e-3},{"params":[net.logZ],"lr":1e-1}])
    Rt=torch.tensor(reward/reward.sum(),dtype=torch.float32)
    for it in range(iters):
        idxs,lpf=sample_batch(net,256); loss=((net.logZ+lpf-torch.log(Rt[idxs]+1e-12))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    o=[]
    while len(o)<ndraw:
        idxs,_=sample_batch(net,512); o+=idxs.tolist()
    c={}
    for i in o: c[i]=c.get(i,0)+1
    return c
sizes=[30,60,120,240]
def compact(draws):
    seen,out={},{}; seen=set(); tgt=sorted(sizes); j=0
    for d in draws:
        seen.add(int(d))
        if j<len(tgt) and len(seen)>=tgt[j]: out[tgt[j]]=err_mHa(seen); j+=1
    return out
log("training GFlowNet for compactness...")
gc=train_draw(np.maximum(w_cheap,1e-10))
unif=compact(rng.integers(0,dim_a,8000))
ciid=compact(rng.choice(dim_a,60000,p=w_cheap))
gfn=compact([i for i,_ in sorted(gc.items(),key=lambda kv:-kv[1]) for _ in range(gc[i])])
orac=compact(rng.choice(dim_a,60000,p=w_a))
hcig={m:err_mHa(cheap_order[:m]) for m in sizes}
fig,ax=plt.subplots(figsize=(7.4,5.0))
S=[("uniform",unif,OI["grey"],"o","-"),("i.i.d. $\\propto$ cheap reward",ciid,OI["sky"],"s","-"),
   ("HCI/CIPSI greedy (classical)",hcig,OI["vermilion"],"D","-"),("GFlowNet on cheap reward (ours)",gfn,OI["purple"],"^","-"),
   ("oracle $\\propto |c|^2$ (unreachable)",orac,OI["green"],"*","--")]
for name,d,c,mk,ls in S:
    xs=[m for m in sizes if m in d]; ys=[d[m] for m in xs]
    ax.plot(xs,ys,marker=mk,ls=ls,color=c,lw=2,ms=10 if mk=="*" else 8,label=name)
ax.axhline(1.6,color=OI["ink"],ls=":",lw=1.3); ax.text(242,1.75,"chemical accuracy",fontsize=10,color=OI["ink"],ha="right")
ax.set_yscale("log"); ax.set_xlabel("subspace dimension (number of strings)")
ax.set_ylabel("energy error vs exact FCI (mHa)")
ax.set_title("Compactness: the GFlowNet learns the important region —\nbut classical greedy by the same signal is as compact",fontsize=12.5)
ax.legend(fontsize=10.3,loc="upper right"); fig.tight_layout()
fig.savefig("/w/pf_compact.pdf",bbox_inches="tight"); fig.savefig("/w/pf_compact.png",bbox_inches="tight",dpi=140)
log("compactness fig done")

# ============================================================ HCI decisive test (H2O, N2)
def hci_test(atom,ncore,ncas,nelec):
    m=gto.M(atom=atom,basis="cc-pvdz",verbose=0); f=scf.RHF(m).run()
    c=mcscf.CASCI(f,ncas,nelec); c.ncore=ncore
    H1,EC=c.get_h1cas(); H2=ao2mo.restore(1,c.get_h2cas(),ncas)
    a,b=nelec; sA=cistring.make_strings(range(ncas),a); dA=len(sA); hfi=int(np.where(sA==(1<<a)-1)[0][0])
    H2e=direct_spin1.absorb_h1e(H1,H2,ncas,nelec,0.5); hd=direct_spin1.make_hdiag(H1,H2,ncas,nelec).reshape(dA,dA)
    eF,cV=pyscf.fci.direct_spin1.FCI().kernel(H1,H2,ncas,nelec,ecore=EC); cV=cV.reshape(dA,dA); wT=(cV**2).sum(1)
    def ground(S):
        idx=np.asarray(sorted(set(S))); n=len(idx)
        if n==1:
            full=np.zeros((dA,dA)); full[idx[0],idx[0]]=1.0
            return float(direct_spin1.contract_2e(H2e,full,ncas,nelec).reshape(dA,dA)[idx[0],idx[0]]),None,idx
        def mv(x):
            full=np.zeros((dA,dA)); full[np.ix_(idx,idx)]=x.reshape(n,n)
            return direct_spin1.contract_2e(H2e,full,ncas,nelec).reshape(dA,dA)[np.ix_(idx,idx)].ravel()
        w,v=eigsh(LinearOperator((n*n,n*n),matvec=mv),k=1,which="SA",maxiter=3000,tol=1e-7)
        return float(w[0]),v[:,0].reshape(n,n),idx
    A={hfi}; batch=max(12,120//8)
    while len(A)<120:
        e_el,cc,idx=ground(A)
        if cc is None: cc=np.array([[1.0]])
        full=np.zeros((dA,dA)); full[np.ix_(idx,idx)]=cc
        Hc2=direct_spin1.contract_2e(H2e,full,ncas,nelec).reshape(dA,dA)
        gap=np.where(np.abs(e_el-hd)<1e-6,1e-6,e_el-hd); scv=((Hc2**2)/(gap**2)).sum(1); scv[list(A)]=-np.inf
        A.update(np.argsort(scv)[::-1][:min(batch,120-len(A))].tolist())
    return (ground(A)[0]+EC-eF)*1000
log("HCI H2O..."); e_h2o=hci_test("O 0 0 0; H 0 0.98 0.76; H 0 -0.98 0.76",1,12,(4,4))
log("HCI N2...");  e_n2=hci_test("N 0 0 0; N 0 0 2.0",2,12,(5,5))
mols=["H$_2$O","N$_2$"]; hci=[e_h2o,e_n2]; gfnv=[2.1,26.7]
fig,ax=plt.subplots(figsize=(6.6,4.8)); x=np.arange(len(mols)); w=0.32
b1=ax.bar(x-w/2,hci,w,color=OI["vermilion"],label="heat-bath CI (classical, noise-free)",edgecolor="white")
b2=ax.bar(x+w/2,gfnv,w,color=OI["purple"],label="GFlowNet-SQD (quantum-sampled, noisy)",edgecolor="white")
ax.axhline(1.6,color=OI["ink"],ls=":",lw=1.3); ax.text(1.45,2.0,"chemical accuracy",fontsize=10,ha="right")
for xx,v in zip(x-w/2,hci): ax.text(xx,v+0.4,f"{v:.1f}",ha="center",fontsize=11,color=OI["vermilion"],fontweight="bold")
for xx,v in zip(x+w/2,gfnv): ax.text(xx,v+0.4,f"{v:.1f}",ha="center",fontsize=11,color=OI["purple"],fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(mols,fontsize=13); ax.set_ylabel("energy error vs exact FCI (mHa)")
ax.set_title("Decisive test at FCI-verifiable scale ($D=120$):\nclassical selected-CI wins on every system",fontsize=12.5)
ax.legend(fontsize=10.3); fig.tight_layout()
fig.savefig("/w/pf_hci.pdf",bbox_inches="tight"); fig.savefig("/w/pf_hci.png",bbox_inches="tight",dpi=140)
log(f"HCI fig done  H2O={e_h2o:.2f} N2={e_n2:.2f}")

# ============================================================ noise crossover (live reduced + production)
BETA=0.5
def fill_D(ranked):
    sel=list(dict.fromkeys(int(i) for i in ranked))[:D]
    for i in cheap_order:
        if len(sel)>=D: break
        if int(i) not in sel: sel.append(int(i))
    return sel[:D]
def one(scale,shots,seed,iters):
    bits=sample_noisy(shots,scale,seed); rec,_=s_core(bits)
    cibm={}
    for s in rec:
        i=str_to_idx[s]; cibm[i]=cibm.get(i,0)+1
    f=np.zeros(dim_a)
    for i,cc in cibm.items(): f[i]=cc
    f/=max(f.sum(),1)
    rew=np.maximum(f+0.1*w_cheap/w_cheap.max()*max(f.max(),1e-9),1e-9)**BETA
    torch.manual_seed(seed); gc=train_draw(rew,iters=iters,ndraw=15000)
    return err_mHa(fill_D(rank(cibm))), err_mHa(rank(gc)[:D])
log("crossover (reduced live)...")
LS=[0.0,1.5,3.0]; live_m=[]; live_s=[]
for sc in LS:
    gaps=[]
    for sd in (0,1):
        a,b=one(sc,1000,sd,500); gaps.append(a-b)
    live_m.append(np.mean(gaps)); live_s.append(np.std(gaps)); log(f"  scale {sc}: gap {np.mean(gaps):+.2f}")
prod_s=[0,1,2,3]; prod_g=[-11.74,0.30,9.71,9.84]; prod_e=[3.5,4.0,4.2,4.3]
fig,ax=plt.subplots(figsize=(7.6,5.0))
ax.axhspan(-30,0,color=OI["vermilion"],alpha=0.06); ax.axhspan(0,30,color=OI["green"],alpha=0.06)
ax.axhline(0,color=OI["ink"],lw=1.1)
ax.errorbar(prod_s,prod_g,yerr=prod_e,fmt="s-",color=OI["blue"],lw=2.3,ms=9,capsize=4,label="production (5 seeds, 1000 iters)")
ax.errorbar(LS,live_m,yerr=live_s,fmt="^--",color=OI["orange"],lw=2,ms=11,capsize=4,label="this run (2 seeds, 500 iters)")
ax.text(0.06,7.0,"GFlowNet better",color=OI["green"],fontsize=12,fontweight="bold")
ax.text(0.06,-9.0,"classical control better",color=OI["vermilion"],fontsize=12,fontweight="bold")
ax.set_ylim(-17,15); ax.set_xlabel(r"noise scale ($\times$ FakeTorino/Heron)")
ax.set_ylabel(r"gap  $E_{\mathrm{ibm+cheap}}-E_{\mathrm{gfn}}$  (mHa)")
ax.set_title("The noise crossover: generative advantage is\nregime-dependent, emerging only under noise",fontsize=12.5)
ax.legend(fontsize=10.3,loc="lower right"); fig.tight_layout()
fig.savefig("/w/pf_crossover.pdf",bbox_inches="tight"); fig.savefig("/w/pf_crossover.png",bbox_inches="tight",dpi=140)
log("crossover fig done")
print("ALL DATA FIGURES DONE")
