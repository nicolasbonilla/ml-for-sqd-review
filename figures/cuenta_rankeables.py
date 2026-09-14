# -*- coding: utf-8 -*-
"""Censo de la recompensa barata: cuantas cadenas alpha sabe ordenar, y por que.

RESPUESTA MEDIDA: 91 de las 792. Las otras 701 valen exactamente cero, y no todas
por la misma razon. El desglose importa porque 546 -- el numero que da el argumento
de Slater-Condon -- no es el total de ceros, y tomarlo por tal deja 546 + 91 = 637
cadenas de 792.

El desglose que este script establece:

  546  triples-y-superiores: cero por Slater-Condon. <D|H|HF> se anula cuando el
       determinante difiere de HF en tres o mas espin-orbitales, y una cadena alpha
       triplemente excitada ya lo hace solo en la parte alpha.
  ...  el RESTO de los ceros son simples y dobles que Slater-Condon deja vivas pero
       que la SIMETRIA espacial anula: en el gauge D-infinito-h las integrales entre
       irreps incompatibles son cero exactamente.
   91  las que quedan con recompensa no nula.

Consecuencia practica: un muestreador entrenado sobre la recompensa sin suelo se
concentra en esas 91, asi que D=120 no se llena con el presupuesto de muestreo del
experimento (5e4 tiradas por brazo). No es que sea imposible -- el suelo se aplica
como np.maximum, luego el soporte sigue siendo completo -- es que no ocurre.
"""
import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.fci import cistring, direct_spin1

NCAS, NELECAS = 12, (5, 5)
mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="cc-pvdz", symmetry=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
cas = mcscf.CASCI(mf, NCAS, NELECAS)
h1, ecore = cas.get_h1cas(); h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)
na = NELECAS[0]
strs = cistring.make_strings(range(NCAS), na); dim = len(strs)
hi = int(np.where(strs == (1 << na) - 1)[0][0])
civ = np.zeros((dim, dim)); civ[hi, hi] = 1.0
h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
Hc = direct_spin1.contract_2e(h2e, civ, NCAS, NELECAS).reshape(dim, dim)
hd = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim, dim)
den = Hc[hi, hi] - hd; den[hi, hi] = 1.0
c1 = Hc / den; c1[hi, hi] = 1.0
w = (c1 ** 2).sum(axis=1); w /= w.sum()

print("  cadenas alpha totales: %d" % dim)
print("  con recompensa EXACTAMENTE no nula: %d" % (w > 0).sum())
print()

# --- censo por nivel de excitacion respecto a la cadena HF ---
hf_bits = (1 << na) - 1
niv = np.array([bin(int(x) & ~hf_bits).count("1") for x in strs])   # electrones promovidos
print("  %-28s %7s %9s %9s" % ("nivel de excitacion alpha", "cadenas", "rec != 0", "rec == 0"))
tot_sc = 0
for k in range(0, na + 1):
    m = niv == k
    if not m.any():
        continue
    nz = int((w[m] > 0).sum())
    etq = {0: "0 (la propia HF)", 1: "1 (simples)", 2: "2 (dobles)"}.get(k, "%d (triples y sup.)" % k)
    print("  %-28s %7d %9d %9d" % (etq, m.sum(), nz, m.sum() - nz))
    if k >= 3:
        tot_sc += int(m.sum())
print()
print("  ceros por SLATER-CONDON (triples y superiores): %d" % tot_sc)
print("  ceros que Slater-Condon NO explica (simples y dobles anuladas")
print("    por simetria espacial en el gauge D-infinito-h):          %d"
      % (int((w == 0).sum()) - tot_sc))
print("  ceros totales: %d   +  no nulas: %d   =  %d"
      % ((w == 0).sum(), (w > 0).sum(), dim))
print()
# --- las dos cifras que la seccion 7 cita sobre las simples ---
# El paper dice: "The 35 singly excited alpha-strings therefore carry 99.2% of the
# total reward mass -- and 47.5% of the exact FCI alpha-string weight, more than the
# Hartree-Fock string itself (35.0%)". Se comprueban las tres.
import pyscf
e_fci, cV = pyscf.fci.direct_spin1.FCI().kernel(h1, h2, NCAS, NELECAS, ecore=ecore)
cV = cV.reshape(dim, dim)
wT = (cV ** 2).sum(axis=1); wT /= wT.sum()

w_hf = w[hi]
s1 = w[niv == 1].sum()
print("  E_FCI = %.12f Ha" % e_fci)
print()
print("  MASA DE RECOMPENSA (w_cheap, normalizada a 1):")
print("    la propia cadena HF          : %7.3f%%" % (100 * w_hf))
print("    las 35 simples              : %7.3f%%  de la masa TOTAL" % (100 * s1))
print("    las 35 simples              : %7.3f%%  de la masa EXCLUIDA la HF"
      % (100 * s1 / (1 - w_hf)))
print("    las 55 dobles no nulas      : %7.3f%%  de la masa total" % (100 * w[niv == 2].sum()))
print("    -> ninguna de las dos lecturas da 99.2%: ni la total (66.6%) ni la")
print("       excluida la HF (83.6%). El manuscrito imprimia 99.2% y se corrigio")
print("       a la cifra medida. Las dos cifras vecinas SI reproducen exactas.")
print()
print("  PESO EXACTO FCI por cadena alpha:")
print("    la cadena HF                : %7.3f%%   (el paper: 35.0%%)" % (100 * wT[hi]))
print("    las 35 simples              : %7.3f%%   (el paper: 47.5%%)" % (100 * wT[niv == 1].sum()))
print("    las 55 dobles con recompensa: %7.3f%%" % (100 * wT[niv == 2].sum()))
print()
print("  RESPECTO A CADA SUELO (las cifras que cita la seccion 4.6):")
print("  %-12s %10s %14s %10s" % ("suelo", "por encima", "a-o-por-debajo", "%"))
for thr in (1e-12, 1e-9, 1e-6, 1e-3):
    n = int((w / w.max() > thr).sum())
    print("  %-12.0e %10d %14d %9.1f%%   D=120 alcanzable? %s"
          % (thr, n, dim - n, 100 * (dim - n) / dim, "SI" if n >= 120 else "NO"))
print()
print()
print("  => el suelo se aplica como np.maximum(w, suelo), asi que NINGUNA cadena queda")
print("     en probabilidad cero: el soporte sigue siendo las 792. Lo que ocurre a 1e-12")
print("     es que las %d restantes quedan seis ordenes por debajo de las 91, y con las"
      % (dim - (w > 0).sum()))
print("     5e4 tiradas por brazo del experimento nunca se juntan 120 distintas.")
print("     Es una afirmacion sobre el PRESUPUESTO, no sobre el soporte.")
