# -*- coding: utf-8 -*-
"""¿Es cierto que un subespacio PRODUCTO A x B necesita |S| >= r_Schmidt(p)^2 ?

Afirmacion: el peso capturado por P_A C P_B cumple
    ||P_A C P_B||_F^2 <= suma de los min(|A|,|B|) mayores valores singulares^2
luego alcanzar peso p exige min(|A|,|B|) >= r(p), y |S|=|A||B| >= r(p)^2.
Se intenta ROMPER con subespacios aleatorios, no solo con los top-k.
"""
import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol=1e-12; mf.run()
cas = mcscf.CASCI(mf, 12, (5,5))
h1, ec = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), 12)
s = fci.direct_spin1.FCI(); s.conv_tol=1e-13
e, C = s.kernel(h1, h2, 12, (5,5))
print(f"E_FCI = {e+ec:.12f}   C shape {C.shape}")

lam = np.sort(np.linalg.svd(C, compute_uv=False)**2)[::-1]
cl = np.cumsum(lam)
r = {p: int(np.searchsorted(cl, p)+1) for p in (0.90,0.99,0.999)}
print(f"rango de Schmidt: r(90%)={r[0.90]}  r(99%)={r[0.99]}  r(99.9%)={r[0.999]}")
print(f"=> cota inferior de DETERMINANTES en cualquier base: r^2 = "
      f"{r[0.90]**2} / {r[0.99]**2} / {r[0.999]**2}\n")

# 1) la cota como desigualdad, sobre subespacios producto ALEATORIOS
rng = np.random.default_rng(11); viol = 0; peor = 0.0
for _ in range(4000):
    na = rng.integers(1, 40); nb = rng.integers(1, 40)
    A = rng.choice(792, na, replace=False); B = rng.choice(792, nb, replace=False)
    cap = float((C[np.ix_(A,B)]**2).sum())
    bound = float(cl[min(na,nb)-1])
    if cap > bound + 1e-12: viol += 1; peor = max(peor, cap-bound)
print(f"1) subespacios producto ALEATORIOS: {viol} violaciones de 4000  (peor exceso {peor:.2e})")

# 2) y con los MEJORES subespacios producto (top-k por peso marginal)
wa = (C**2).sum(1); wb = (C**2).sum(0)
oa = np.argsort(-wa); ob = np.argsort(-wb)
viol2 = 0
print("\n2) mejores subespacios top-k:  k   peso capturado   cota r^2      ¿cumple?")
for k in (5, 9, 12, 15, 21, 30):
    A = oa[:k]; B = ob[:k]
    cap = float((C[np.ix_(A,B)]**2).sum())
    bound = float(cl[k-1])
    ok = cap <= bound + 1e-12
    viol2 += (not ok)
    print(f"   {k:3d}   {cap:.6f}      {bound:.6f}    {'si' if ok else 'NO'}")
print(f"\n   violaciones: {viol2}")

# 3) ¿es alcanzable la cota? el minimo |S| real para 90% con producto top-k
for k in range(1, 60):
    cap = float((C[np.ix_(oa[:k], ob[:k])]**2).sum())
    if cap >= 0.90:
        print(f"\n3) minimo producto top-k que alcanza 90%: k={k} -> |S|={k*k} determinantes")
        print(f"   cota inferior invariante r^2 = {r[0.90]**2}   (holgura {k*k - r[0.90]**2})")
        break
