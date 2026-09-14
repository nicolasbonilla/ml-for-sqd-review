"""Second FCI-verifiable system for the noise crossover: H2O CAS(8e,12o) cc-pVDZ (equilibrium,
single-reference). Same pipeline as the N2 sweep: per noise scale, gap between the fair classical
control (S-CORE recovery + cheap prior) and the fused generative proposer, 5 seeds. Tests whether
the crossover is system-general (shot-starvation) or specific to the multireference regime."""
import time, json
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1
t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

LAMBDA0,P100,P010=0.05,0.0229,0.0200
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    pr=FakeTorino().properties(); nq=FakeTorino().num_qubits
    P100=float(np.median([pr.qubit_property(q,"prob_meas0_prep1")[0] for q in range(nq)]))
    P010=float(np.median([pr.qubit_property(q,"prob_meas1_prep0")[0] for q in range(nq)]))
except Exception: pass

# equilibrium H2O, CAS(8e,12o)
NCAS,NELECAS=12,(4,4); na,nb=NELECAS
# symmetry=True fija el gauge orbital. Sin esto los orbitales canonicos RHF pueden
# rotar entre corridas dentro de capas degeneradas a energia identica, y las barras
# de error mezclan ruido del muestreador con ruido de gauge (vease gauge_study/).
mol=gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",basis="cc-pvdz",symmetry=True,verbose=0)
mf=scf.RHF(mol).run()
cas=mcscf.CASCI(mf,NCAS,NELECAS)
h1,ecore=cas.get_h1cas(); h2=ao2mo.restore(1,cas.get_h2cas(),NCAS)
strs_a=cistring.make_strings(range(NCAS),na); dim_a=len(strs_a)
str_to_idx={int(s):i for i,s in enumerate(strs_a)}
set_to_idx={frozenset(p for p in range(NCAS) if (int(s)>>p)&1):i for i,s in enumerate(strs_a)}
HF=(1<<na)-1; hf_idx=str_to_idx[HF]
civ_hf=np.zeros((dim_a,dim_a)); civ_hf[hf_idx,hf_idx]=1.0
h2e=direct_spin1.absorb_h1e(h1,h2,NCAS,NELECAS,0.5)
Hc=direct_spin1.contract_2e(h2e,civ_hf,NCAS,NELECAS).reshape(dim_a,dim_a)
hd=direct_spin1.make_hdiag(h1,h2,NCAS,NELECAS).reshape(dim_a,dim_a)
den=Hc[hf_idx,hf_idx]-hd; den[hf_idx,hf_idx]=1.0
c1=Hc/den; c1[hf_idx,hf_idx]=1.0
w_cheap=(c1**2).sum(1); w_cheap/=w_cheap.sum(); cheap_order=np.argsort(w_cheap)[::-1]
e_fci,civec=pyscf.fci.direct_spin1.FCI().kernel(h1,h2,NCAS,NELECAS,ecore=ecore)
civec=civec.reshape(dim_a,dim_a); w_true=(civec**2).sum(1); w_true/=w_true.sum()
w_hf=float(civec[hf_idx,hf_idx]**2)
log(f"H2O E_FCI={e_fci:.6f}  dim_a={dim_a}  HF weight={w_hf:.3f} (single-reference if high)")
_sci=selected_ci.SelectedCI()
def E(idxs):
    s=np.asarray(sorted(set(int(strs_a[i]) for i in idxs)),dtype=np.int64)
    out=selected_ci.kernel_fixed_space(_sci,h1,h2,NCAS,NELECAS,(s,s),ecore=ecore)
    return (float(out[0] if isinstance(out,(tuple,list)) else out)-e_fci)*1000
class Policy(nn.Module):
    def __init__(s,n):
        super().__init__(); s.net=nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,n)); s.logZ=nn.Parameter(torch.zeros(1))
    def forward(s,x): return s.net(x)
def sb(net,B):
    st=torch.zeros(B,NCAS); lpf=torch.zeros(B)
    for _ in range(na):
        lp=torch.log_softmax(net(st).masked_fill(st.bool(),-1e9),1)
        a=torch.distributions.Categorical(logits=lp).sample(); lpf+=lp.gather(1,a[:,None]).squeeze(1); st=st.scatter(1,a[:,None],1.0)
    return np.array([set_to_idx[frozenset(np.where(st[b].numpy()>0)[0].tolist())] for b in range(B)]),lpf
def tad(reward,ndraw=15000,iters=1000):
    net=Policy(NCAS); opt=torch.optim.Adam([{"params":net.net.parameters(),"lr":1e-3},{"params":[net.logZ],"lr":1e-1}])
    Rt=torch.tensor(reward/reward.sum(),dtype=torch.float32)
    for it in range(iters):
        idxs,lpf=sb(net,256); loss=((net.logZ+lpf-torch.log(Rt[idxs]+1e-12))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    o=[]
    while len(o)<ndraw:
        idxs,_=sb(net,512); o+=idxs.tolist()
    c={}
    for i in o: c[i]=c.get(i,0)+1
    return c
def b2s(b): return int(sum(int(v)<<p for p,v in enumerate(b)))
D=120; BETA_T=0.5
def fill(ranked):
    sel=list(dict.fromkeys(ranked))[:D]
    for i in cheap_order:
        if len(sel)>=D: break
        if i not in sel: sel.append(int(i))
    return sel[:D]
def one(scale,shots,seed):
    P10,P01,LAM=min(P100*scale,0.5),min(P010*scale,0.5),min(LAMBDA0*scale,0.9)
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    ideal=rng.choice(dim_a,size=shots,p=w_true)
    bits=np.array([[(int(strs_a[i])>>p)&1 for p in range(NCAS)] for i in ideal],dtype=np.int8)
    dep=rng.random(shots)<LAM; bits[dep]=(rng.random((dep.sum(),NCAS))<0.5).astype(np.int8)
    o,z=bits==1,bits==0; bits[o&(rng.random(bits.shape)<P10)]=0; bits[z&(rng.random(bits.shape)<P01)]=1
    good=np.array([int(b.sum())==na for b in bits]); occ=bits[good].mean(0) if good.any() else np.full(NCAS,na/NCAS)
    rec=[]
    for row in bits:
        s=row.copy(); m=int(s.sum())
        if m==na: rec.append(b2s(s)); continue
        if m>na:
            od=np.where(s==1)[0]; s[od[np.argsort(occ[od])[:m-na]]]=0
        else:
            em=np.where(s==0)[0]; s[em[np.argsort(occ[em])[::-1][:na-m]]]=1
        rec.append(b2s(s))
    cibm={}
    for s in rec:
        i=str_to_idx[s]; cibm[i]=cibm.get(i,0)+1
    f=np.zeros(dim_a)
    for i,cc in cibm.items(): f[i]=cc
    f/=max(f.sum(),1)
    rew=np.maximum(f+0.1*w_cheap/w_cheap.max()*max(f.max(),1e-9),1e-9)**BETA_T
    gf=tad(rew); rank=lambda c:[i for i,_ in sorted(c.items(),key=lambda kv:-kv[1])]
    return 100*(1-good.mean()),E(fill(rank(cibm))),E(rank(gf)[:D])

res={"system":"H2O","active":"(8e,12o)","E_FCI":float(e_fci),"HF_weight":w_hf,"D":D,"scales":{}}
SEEDS=[0,1,2,3,4]
log(f"{'scale':>5s} {'lost%':>6s} {'ibm+cheap':>13s} {'gfn-fused':>13s} {'gap±std':>14s}")
for scale in (0.0,1.0,2.0,3.0):
    lost,eibm,egf,gaps=[],[],[],[]
    for sd in SEEDS:
        l,a,b=one(scale,1000,sd); lost.append(l); eibm.append(a); egf.append(b); gaps.append(a-b)
    ga=np.array(gaps); sem=float(np.std(ga,ddof=1)/np.sqrt(len(ga)))
    res["scales"][f"{scale:.0f}"]={"lost_pct":float(np.mean(lost)),
        "ibm_cheap_mean":float(np.mean(eibm)),"ibm_cheap_std":float(np.std(eibm,ddof=1)),
        "gfn_mean":float(np.mean(egf)),"gfn_std":float(np.std(egf,ddof=1)),
        "gap_mean":float(np.mean(ga)),"gap_std":float(np.std(ga,ddof=1)),"gap_sem":sem,
        "t_stat":float(np.mean(ga)/sem) if sem>0 else 0.0}
    log(f"{scale:5.1f} {np.mean(lost):6.1f} {np.mean(eibm):7.2f}±{np.std(eibm,ddof=1):4.1f} "
        f"{np.mean(egf):7.2f}±{np.std(egf,ddof=1):4.1f} {np.mean(ga):7.2f}±{np.std(ga,ddof=1):4.1f}")
json.dump(res,open("/w/h2o_crossover.json","w"),indent=1)
log("WROTE h2o_crossover.json")
