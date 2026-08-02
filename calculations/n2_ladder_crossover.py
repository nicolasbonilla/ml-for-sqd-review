"""CONTROLLED experiment (referee-requested): the noise crossover along the N2 geometry ladder.
Same molecule, same electron count (10e), same active space (12o) and dimension (792 alpha-strings)
at EVERY point -- only the multireference character (bond length R) varies. This removes the
molecule/electron/dimension confound of the N2-vs-H2O contrast: it varies ONLY the tuned variable.
For each R we compute the classical-minus-generative gap at zero and high noise, 5 seeds, and the
Spearman order parameter rho(R). Expectation: the high-noise gap grows as rho falls (multireference)."""
import time, json
import numpy as np
import torch, torch.nn as nn
import pyscf
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, selected_ci, direct_spin1
from scipy.stats import spearmanr
t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

LAMBDA0,P100,P010=0.05,0.0229,0.0200
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    pr=FakeTorino().properties(); nq=FakeTorino().num_qubits
    P100=float(np.median([pr.qubit_property(q,"prob_meas0_prep1")[0] for q in range(nq)]))
    P010=float(np.median([pr.qubit_property(q,"prob_meas1_prep0")[0] for q in range(nq)]))
except Exception: pass

NCAS,NELECAS=12,(5,5); na,nb=NELECAS
GEOMS=[1.10,1.70,2.00,2.50]; SCALES=[0.0,3.0]; SEEDS=[0,1,2,3,4]; ITERS=800; D=120; BETA_T=0.5; SHOTS=1000

class Policy(nn.Module):
    def __init__(s,n):
        super().__init__(); s.net=nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,n)); s.logZ=nn.Parameter(torch.zeros(1))
    def forward(s,x): return s.net(x)

def run_geometry(R):
    mol=gto.M(atom=f"N 0 0 0; N 0 0 {R}",basis="cc-pvdz",verbose=0)
    mf=scf.RHF(mol).run(); cas=mcscf.CASCI(mf,NCAS,NELECAS)
    h1,ecore=cas.get_h1cas(); h2=ao2mo.restore(1,cas.get_h2cas(),NCAS)
    strs_a=cistring.make_strings(range(NCAS),na); dim_a=len(strs_a)
    str_to_idx={int(s):i for i,s in enumerate(strs_a)}
    set_to_idx={frozenset(p for p in range(NCAS) if (int(s)>>p)&1):i for i,s in enumerate(strs_a)}
    HF=(1<<na)-1; hf_idx=str_to_idx[HF]
    civ=np.zeros((dim_a,dim_a)); civ[hf_idx,hf_idx]=1.0
    h2e=direct_spin1.absorb_h1e(h1,h2,NCAS,NELECAS,0.5)
    Hc=direct_spin1.contract_2e(h2e,civ,NCAS,NELECAS).reshape(dim_a,dim_a)
    hd=direct_spin1.make_hdiag(h1,h2,NCAS,NELECAS).reshape(dim_a,dim_a)
    den=Hc[hf_idx,hf_idx]-hd; den[hf_idx,hf_idx]=1.0
    c1=Hc/den; c1[hf_idx,hf_idx]=1.0
    w_cheap=(c1**2).sum(1); w_cheap/=w_cheap.sum(); cheap_order=np.argsort(w_cheap)[::-1]
    e_fci,cV=pyscf.fci.direct_spin1.FCI().kernel(h1,h2,NCAS,NELECAS,ecore=ecore); cV=cV.reshape(dim_a,dim_a)
    w_true=(cV**2).sum(1); w_true/=w_true.sum(); w_hf=float(cV[hf_idx,hf_idx]**2)
    rho=float(spearmanr(w_cheap,w_true).correlation)
    _sci=selected_ci.SelectedCI()
    def E(idxs):
        s=np.asarray(sorted(set(int(strs_a[i]) for i in idxs)),dtype=np.int64)
        o=selected_ci.kernel_fixed_space(_sci,h1,h2,NCAS,NELECAS,(s,s),ecore=ecore)
        return (float(o[0] if isinstance(o,(tuple,list)) else o)-e_fci)*1000
    def sb(net,B):
        st=torch.zeros(B,NCAS); lpf=torch.zeros(B)
        for _ in range(na):
            lp=torch.log_softmax(net(st).masked_fill(st.bool(),-1e9),1)
            a=torch.distributions.Categorical(logits=lp).sample(); lpf+=lp.gather(1,a[:,None]).squeeze(1); st=st.scatter(1,a[:,None],1.0)
        return np.array([set_to_idx[frozenset(np.where(st[k].numpy()>0)[0].tolist())] for k in range(B)]),lpf
    def tad(rew):
        net=Policy(NCAS); opt=torch.optim.Adam([{"params":net.net.parameters(),"lr":1e-3},{"params":[net.logZ],"lr":1e-1}])
        Rt=torch.tensor(rew/rew.sum(),dtype=torch.float32)
        for _ in range(ITERS):
            idx,lpf=sb(net,256); loss=((net.logZ+lpf-torch.log(Rt[idx]+1e-12))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        o=[]
        while len(o)<15000: idx,_=sb(net,512); o+=idx.tolist()
        c={}
        for i in o: c[i]=c.get(i,0)+1
        return c
    def b2s(bb): return int(sum(int(v)<<p for p,v in enumerate(bb)))
    def one(scale,seed):
        P10,P01,LAM=min(P100*scale,0.5),min(P010*scale,0.5),min(LAMBDA0*scale,0.9)
        rng=np.random.default_rng(seed); torch.manual_seed(seed)
        ideal=rng.choice(dim_a,size=SHOTS,p=w_true)
        bits=np.array([[(int(strs_a[i])>>p)&1 for p in range(NCAS)] for i in ideal],dtype=np.int8)
        dep=rng.random(SHOTS)<LAM; bits[dep]=(rng.random((dep.sum(),NCAS))<0.5).astype(np.int8)
        o,z=bits==1,bits==0; bits[o&(rng.random(bits.shape)<P10)]=0; bits[z&(rng.random(bits.shape)<P01)]=1
        good=np.array([int(bb.sum())==na for bb in bits]); occ=bits[good].mean(0) if good.any() else np.full(NCAS,na/NCAS)
        rec=[]
        for row in bits:
            s=row.copy(); m=int(s.sum())
            if m==na: rec.append(b2s(s)); continue
            if m>na: od=np.where(s==1)[0]; s[od[np.argsort(occ[od])[:m-na]]]=0
            else: em=np.where(s==0)[0]; s[em[np.argsort(occ[em])[::-1][:na-m]]]=1
            rec.append(b2s(s))
        cibm={}
        for s in rec: i=str_to_idx[s]; cibm[i]=cibm.get(i,0)+1
        f=np.zeros(dim_a)
        for i,cc in cibm.items(): f[i]=cc
        f/=max(f.sum(),1)
        rew=np.maximum(f+0.1*w_cheap/w_cheap.max()*max(f.max(),1e-9),1e-9)**BETA_T
        gf=tad(rew); rank=lambda c:[i for i,_ in sorted(c.items(),key=lambda kv:-kv[1])]
        sel=list(dict.fromkeys(rank(cibm)))[:D]
        for i in cheap_order:
            if len(sel)>=D: break
            if i not in sel: sel.append(int(i))
        lost=100*(1-good.mean())
        return E(sel[:D]),E(rank(gf)[:D]),lost
    res={"R":R,"rho":rho,"w_hf":w_hf,"scales":{}}
    for scale in SCALES:
        gaps,lost=[],[]
        for sd in SEEDS:
            ei,eg,lo=one(scale,sd); gaps.append(ei-eg); lost.append(lo)
        ga=np.array(gaps); sem=float(np.std(ga,ddof=1)/np.sqrt(len(ga)))
        res["scales"][f"{scale:.0f}"]={"gap_mean":float(np.mean(ga)),"gap_std":float(np.std(ga,ddof=1)),
            "gap_sem":sem,"t_stat":float(np.mean(ga)/sem) if sem>0 else 0.0,"lost_pct":float(np.mean(lost))}
        log(f"  R={R:.2f} scale={scale:.0f}: gap={np.mean(ga):+6.2f}+-{np.std(ga,ddof=1):4.2f} (t={np.mean(ga)/sem if sem>0 else 0:+.1f})")
    log(f"R={R:.2f} DONE  rho={rho:.3f} w_HF={w_hf:.3f}")
    return res

out={"note":"controlled: fixed N2(10e,12o), only geometry varies","GEOMS":GEOMS,"SCALES":SCALES,"SEEDS":SEEDS,"geoms":[]}
for R in GEOMS: out["geoms"].append(run_geometry(R))
json.dump(out,open("/w/n2_ladder_crossover.json","w"),indent=1)
# dat for the figure: R rho w_hf gap0 gap0_sd gap3 gap3_sd
with open("/w/ladder.dat","w") as fh:
    fh.write("R rho whf gap0 gap0sd gap3 gap3sd t3\n")
    for g in out["geoms"]:
        s0=g["scales"]["0"]; s3=g["scales"]["3"]
        fh.write(f"{g['R']:.2f} {g['rho']:.4f} {g['w_hf']:.4f} {s0['gap_mean']:.3f} {s0['gap_std']:.3f} {s3['gap_mean']:.3f} {s3['gap_std']:.3f} {s3['t_stat']:.3f}\n")
log("WROTE n2_ladder_crossover.json + ladder.dat")
