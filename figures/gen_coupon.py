# -*- coding: utf-8 -*-
"""Genera coupon_w.dat y coupon_cum.dat para la Fig. 3.

POR QUE EXISTE ESTE FICHERO. Hasta el 2026-09-10 ningun script del repositorio
producia esos dos .dat: eran artefactos de un scratchpad de Docker ya borrado,
sin energia de referencia dentro, y por tanto no auditables. Eso dejo abierta
durante dias una disputa sobre si el 90% del peso vivia en 11 o en 12 cadenas,
que resulto no ser un error de nadie sino dependencia de gauge.

QUE HACE DISTINTO. Fija el gauge explicitamente con simetria de punto D-infinity-h
(mol.symmetry=True), que es reproducible: los orbitales canonicos SIN simetria no
lo son -- cinco corridas del mismo codigo devuelven cinco orientaciones distintas
de las capas pi degeneradas, con la misma energia a 1e-13 Ha. Y escribe E_FCI y
la receta completa en la cabecera de cada fichero, para que nunca vuelva a haber
un .dat que no se pueda comprobar.

Uso:  python gen_coupon.py            (escribe en el directorio actual)

La cabecera va comentada con %% porque es lo que pgfplots entiende de forma
nativa (con # lee la cabecera como datos y destroza la grafica, sin dar error).
Desde numpy:  np.loadtxt(fichero, comments=chr(37), skiprows=8)
(7 lineas de cabecera comentada + 1 de nombres de columna; comments= sola no basta
porque la linea de nombres no lleva marca de comentario)
"""
import numpy as np
from pyscf import ao2mo, gto, mcscf, scf, symm
from pyscf.fci import cistring, direct_spin1

NCAS, NELECAS = 12, (5, 5)
na, nb = NELECAS
GEOM = "N 0 0 0; N 0 0 2.0"
BASIS = "cc-pvdz"
SCF_TOL, FCI_TOL = 1e-12, 1e-13

mol = gto.M(atom=GEOM, basis=BASIS, symmetry=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = SCF_TOL
mf.run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas()
h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
solver = direct_spin1.FCI()
solver.conv_tol = FCI_TOL
e, civ = solver.kernel(h1, h2, NCAS, NELECAS)
e_fci = float(e + ecore)

irreps = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
activos = " ".join(str(irreps[i]) for i in range(cas.ncore, cas.ncore + NCAS))

strs_a = cistring.make_strings(range(NCAS), na)
w = np.sort((civ ** 2).sum(axis=1))[::-1]
cum = np.cumsum(w)

CAB = (
    "% N2 {geom} / {basis} / CASCI({ne}e,{no}o) ncore={nc}\n"
    "% gauge: orbitales RHF adaptados por simetria, grupo {gr}\n"
    "% irreps del espacio activo: {ir}\n"
    "% scf.conv_tol={st:g}  fci.conv_tol={ft:g}\n"
    "% E_RHF = {erhf:.12f} Ha\n"
    "% E_FCI = {efci:.12f} Ha\n"
    "% suma de pesos = {sw:.12f}   ({n} cadenas alpha)\n"
).format(geom=GEOM, basis=BASIS, ne=na + nb, no=NCAS, nc=cas.ncore,
         gr=mol.groupname, ir=activos, st=SCF_TOL, ft=FCI_TOL,
         erhf=mf.e_tot, efci=e_fci, sw=float(w.sum()), n=len(w))

with open("coupon_w.dat", "w") as fh:
    fh.write(CAB + "rank weight\n")
    for i, v in enumerate(w, 1):
        fh.write("%d %.6e\n" % (i, v))

with open("coupon_cum.dat", "w") as fh:
    fh.write(CAB + "n cum\n")
    for i, v in enumerate(cum, 1):
        fh.write("%d %.5f\n" % (i, v))

print("E_FCI = %.12f Ha" % e_fci)
print("grupo = %s   irreps activos = %s" % (mol.groupname, activos))
for thr, lab in ((0.90, "90%"), (0.99, "99%"), (0.999, "99.9%")):
    i = int(np.searchsorted(cum, thr) + 1)
    k = i
    while k < len(w) and abs(w[k] - w[k - 1]) <= 1e-6 * abs(w[k - 1]):
        k += 1
    print("  %6s: cruce n=%d -> multiplete completo n=%d  (cum=%.5f, %.2f%% de %d)"
          % (lab, i, k, cum[k - 1], 100.0 * k / len(w), len(w)))
print("escritos coupon_w.dat y coupon_cum.dat, con E_FCI en la cabecera")
