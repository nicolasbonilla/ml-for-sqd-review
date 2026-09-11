# -*- coding: utf-8 -*-
r"""Genera wf_occ.dat y wf_hist.dat de la Fig. 1 (fig:loop) de P1.

RECETA DECLARADA (identica a la de todos los demas scripts del repositorio):
    N2, R = 2.0 A, cc-pVDZ, CAS(10e,12o), CASCI sobre orbitales canonicos RHF.
    Reproduce E_FCI = -108.8080419 Ha.

POR QUE EXISTE ESTE FICHERO
---------------------------
Los .dat que llevaba la Fig. 1 hasta 2026-09-11 NO salian de esta receta: daban
los pares pi degenerados ROTOS (0.6568 / 0.6509) donde el FCI exacto los da
identicos (0.6603 / 0.6603), y una cola tres veces mayor. Nadie podia
regenerarlos. Se sustituyen por estos, que si salen de la receta que el paper
declara. La Fig. 1 es el esquema del flujo: sus insets ilustran que el peso se
concentra y que hay caracter multirreferencial, y ambas cosas siguen siendo
ciertas -- ahora ademas reproducibles.

DETERMINISMO
------------
El histograma usa `default_rng(0)`. Fije OMP_NUM_THREADS=1 si quiere
reproducibilidad bit a bit de las orientaciones orbitales: con capas degeneradas
los orbitales canonicos RHF no son reproducibles entre corridas con hilos
distintos (vease gauge_study/determinismo.py).

USO
---
    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci python /w/gen_fig1.py
"""
import numpy as np
from pyscf import fci, gto, mcscf, scf

NCAS, NELECAS = 12, (5, 5)
SHOTS = 3000
SEMILLA = 0

mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", verbose=0)
mf = scf.RHF(mol).run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
e_fci = cas.kernel()[0]
print("RHF    E = %.9f Ha" % mf.e_tot)
print("CASCI  E = %.12f Ha" % e_fci)
assert abs(e_fci + 108.808041914843) < 1e-7, "la receta no reproduce E_FCI"

ci = np.asarray(cas.ci)

# --- ocupaciones: diagonal del 1-RDM alpha en la base MO canonica del espacio activo
dm1a, _ = cas.fcisolver.make_rdm1s(ci, NCAS, NELECAS)
occ = np.diag(dm1a).real
with open("/w/wf_occ.dat", "w") as f:
    f.write("%% ocupaciones del espacio activo -- diagonal del 1-RDM alpha, base MO canonica\n")
    f.write("%% N2 R=2.0A cc-pVDZ CAS(10e,12o) CASCI   E_FCI = %.12f Ha\n" % e_fci)
    f.write("%% generado por gen_fig1.py ; suma = %.6f (= 5 electrones alpha)\n" % occ.sum())
    f.write("orb occ\n")
    for i, x in enumerate(occ):
        f.write("%d %.4f\n" % (i, x))
print("\nwf_occ.dat:  " + "  ".join("%.4f" % x for x in occ))
print("             suma = %.4f" % occ.sum())

# --- histograma: muestreo Born de las cadenas alpha (panel "3 . Sample bitstrings")
na = fci.cistring.num_strings(NCAS, NELECAS[0])
w = (ci ** 2).sum(axis=1)
w /= w.sum()
cnt = np.random.default_rng(SEMILLA).multinomial(SHOTS, w)
top = np.sort(cnt)[::-1][:14]
with open("/w/wf_hist.dat", "w") as f:
    f.write("%% cuentas de las 14 cadenas alpha mas muestreadas\n")
    f.write("%% muestreo Born de |c_alpha|^2 sobre %d cadenas, %d disparos, semilla %d\n"
            % (na, SHOTS, SEMILLA))
    f.write("%% generado por gen_fig1.py ; N2 R=2.0A cc-pVDZ CAS(10e,12o)\n")
    f.write("rank count\n")
    for i, x in enumerate(top):
        f.write("%d %d\n" % (i, x))
print("\nwf_hist.dat: %s" % list(top))
print("             %d cadenas, %d disparos, semilla %d" % (na, SHOTS, SEMILLA))
print("\nlisto.")
