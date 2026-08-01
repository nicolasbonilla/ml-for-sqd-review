# -*- coding: utf-8 -*-
"""Generate Gaussian cube files for selected active-space MOs + save orbital energies."""
import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen

def do(tag, atom, picks):
    mol = gto.M(atom=atom, basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    np.save(f"/w/{tag}_moe.npy", mf.mo_energy)
    np.save(f"/w/{tag}_nocc.npy", np.array([nocc]))
    for idx, name in picks:
        cubegen.orbital(mol, f"/w/{tag}_{name}.cube", mf.mo_coeff[:, idx],
                        nx=90, ny=90, nz=90)
        print(f"{tag} {name}: MO#{idx}  e={mf.mo_energy[idx]:+.3f} Ha", flush=True)
    return mf.mo_energy, nocc

# N2 (nocc=7): HOMO=6 (3sigma_g), HOMO-1/2 = 1pi_u (deg, 4,5), LUMO=7/8 (1pi_g*)
do("n2", "N 0 0 0; N 0 0 2.0",
   [(5, "piu_homo1"), (6, "sg_homo"), (7, "pig_lumo"), (9, "su_lumo3")])
# H2O (nocc=5): HOMO=4 (1b1 lone pair), LUMO=5 (4a1*)
do("h2o", "O 0 0 0; H 0 0.98 0.76; H 0 -0.98 0.76",
   [(3, "a1_homo1"), (4, "b1_homo"), (5, "a1_lumo")])
print("cubes done")
