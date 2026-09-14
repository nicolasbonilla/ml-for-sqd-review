# -*- coding: utf-8 -*-
"""Orientacion de la capa pi degenerada, corrida a corrida.
Sonda: angulo del MO5 en el plano 2px/2py del PRIMER N.
(las etiquetas AO de pyscf llevan espacios finales -> hay que hacer strip)"""
import numpy as np, sys
from pyscf import gto, scf
# Sin argumento corre el sorteo canonico, que es el caso interesante. Antes exigia
# el argumento y reventaba con IndexError si se corria como lo documenta el README.
sym = len(sys.argv) > 1 and sys.argv[1] == "sym"
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", symmetry=sym, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
L = [l.strip() for l in mol.ao_labels()]
ix = L.index('0 N 2px'); iy = L.index('0 N 2py')
cx, cy = mf.mo_coeff[ix, 5], mf.mo_coeff[iy, 5]
print(f"{'SIM ' if sym else 'CANO'}  E_RHF={mf.e_tot:.14f}  "
      f"angulo={np.degrees(np.arctan2(cy,cx))%180.0:8.3f}deg  cx={cx:+.9f} cy={cy:+.9f}")
