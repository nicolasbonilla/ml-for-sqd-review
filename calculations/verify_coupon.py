# -*- coding: utf-8 -*-
"""Independently recompute the coupon-collector number for N2 CAS(10e,12o) cc-pVDZ.
How many single-spin (alpha) strings carry 90% of the exact FCI weight, out of C(12,5)=792?"""
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
from math import comb

for R in (2.0,):  # Angstrom, stretched triple bond
    # symmetry=True NO es cosmetico aqui: este script verifica el conteo de cadenas, que
    # es justo la cantidad que depende del gauge. Sin fijarlo da 12 al 90% mientras el
    # fichero depositado (coupon_cum.dat, gauge D-infinito-h) cruza en 11. Correr con
    # OMP_NUM_THREADS=1.
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", spin=0,
                symmetry=True, verbose=0)
    mf = scf.RHF(mol).run()
    ncas, nelecas = 12, 10           # CAS(10e,12o)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    h1, ecore = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), ncas)
    na = nelecas // 2                 # 5 alpha electrons
    ndet_str = comb(ncas, na)         # C(12,5) = 792
    cisolver = fci.direct_spin1.FCI()
    e, civec = cisolver.kernel(h1, h2, ncas, (na, na), ecore=ecore)
    # civec shape = (n_alpha_strings, n_beta_strings)
    civec = np.asarray(civec)
    # marginal weight of each alpha string = sum over beta of |c|^2
    w_alpha = (civec**2).sum(axis=1)
    w_alpha_sorted = np.sort(w_alpha)[::-1]
    cum = np.cumsum(w_alpha_sorted)
    # smallest number of strings whose cumulative weight >= 0.90
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    total_strings = civec.shape[0]
    print(f"R={R} A  E_FCI = {e:.6f} Ha")
    print(f"alpha strings total = {total_strings}  (C(12,5) = {ndet_str})")
    print(f"strings carrying >=90% weight: {n90}  ({100*n90/total_strings:.2f}% of {total_strings})")
    print(f"  top-1 weight = {w_alpha_sorted[0]:.4f}")
    print(f"  cum@10 = {cum[9]:.4f}  cum@11 = {cum[10]:.4f}  cum@12 = {cum[11]:.4f}  cum@13 = {cum[12]:.4f}")
