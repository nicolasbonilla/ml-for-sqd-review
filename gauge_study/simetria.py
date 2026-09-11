# -*- coding: utf-8 -*-
"""Etiquetas de simetria de las tres capas degeneradas del espacio activo."""
import numpy as np
from pyscf import gto, scf, symm

mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol).run()
labels = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
print(f"grupo puntual detectado por pyscf: {mol.groupname}")
print(f"E_RHF = {mf.e_tot:.9f}\n")
print(" MO   energia(Ha)   energia(eV)   irrep   ocup")
for i in range(0, 14):
    print(f" {i:3d}  {mf.mo_energy[i]:12.8f}  {mf.mo_energy[i]*27.211386:10.4f}"
          f"   {labels[i]:>6s}   {mf.mo_occ[i]:.0f}")
print("\ncapas degeneradas dentro del activo (MO 2..13), con su irrep:")
i = 2
while i < 14:
    j = i
    while j + 1 < 14 and abs(mf.mo_energy[j+1] - mf.mo_energy[i]) < 1e-6:
        j += 1
    if j > i:
        print(f"  MO {list(range(i,j+1))}  E={mf.mo_energy[i]:.8f} Ha"
              f"  irreps={[labels[k] for k in range(i,j+1)]}")
    i = j + 1
