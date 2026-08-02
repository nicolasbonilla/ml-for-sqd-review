# -*- coding: utf-8 -*-
"""Generate the cube for the GENUINE 3sigma_g orbital of N2 (MO#4, -13.0 eV, non-degenerate).
The earlier n2_sg_homo.cube was MO#6 = a 1pi_u component (mislabelled); this fixes it."""
import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
for idx, name in [(4, "sg_true")]:
    cubegen.orbital(mol, f"/w/n2_{name}.cube", mf.mo_coeff[:, idx], nx=90, ny=90, nz=90)
    print(f"n2 {name}: MO#{idx}  e={mf.mo_energy[idx]*27.211386:+.3f} eV", flush=True)
print("done")
