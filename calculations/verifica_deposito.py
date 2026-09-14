"""Check the numbers PRINTED IN THE MANUSCRIPT against the deposited data files.

Section 6 of the paper asks that a deposit be able to regenerate the figures it
accompanies. That is element (x), and it is the one the paper's own self-audit caught
itself failing. This script closes the loop mechanically: it reads main.tex, pulls out
the values actually typeset, reads the .dat and .json files, and compares them.

It does NOT hold a second copy of the numbers. Every expected value is computed from a
deposited file at run time, and every printed value is extracted from main.tex at run
time. A number can therefore only pass if the manuscript and the deposit agree, which
is the property that kept failing by hand: a figure would be regenerated and the
sentence describing it left alone, or the reverse.

Exit status is 0 if every check passes and 1 otherwise, so it can be run in CI or as
the last step before a submission.

    docker run --rm -v "$PWD":/w -w /w sqd-fci python /w/calculations/verifica_deposito.py

No PySCF is required -- it only reads files.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
TEX = io.open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()

fallos = []
pasan = 0


def dat(rel):
    """Read a '%'-commented .dat into {column: [floats]}."""
    rows = [l.split() for l in io.open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8")
            if l.strip() and not l.lstrip().startswith("%")]
    return {h: [float(r[k]) for r in rows[1:]] for k, h in enumerate(rows[0])}


def jsn(rel):
    return json.load(io.open(os.path.join(ROOT, *rel.split("/")), encoding="utf-8"))


def impreso(patron, etiqueta):
    """Pull the value(s) the manuscript actually typesets. Fails loudly if absent."""
    m = re.search(patron, TEX)
    if not m:
        fallos.append((etiqueta, "el patron no aparece en main.tex -- la frase cambio"))
        return None
    return [float(g.replace("{,}", ".")) for g in m.groups()]


def comprueba(etiqueta, patron, esperados, tol=0.051):
    """esperados: the values the deposit says, in the order the regex captures them."""
    global pasan
    vals = impreso(patron, etiqueta)
    if vals is None:
        return
    if len(vals) != len(esperados):
        fallos.append((etiqueta, "capture %d valores y el deposito da %d"
                       % (len(vals), len(esperados))))
        return
    malos = [(a, b) for a, b in zip(vals, esperados) if abs(a - b) > tol]
    if malos:
        fallos.append((etiqueta, "; ".join("el paper dice %.4g, el deposito %.4g" % (a, b)
                                           for a, b in malos)))
    else:
        pasan += 1
        print("  OK  %s" % etiqueta)


print("=" * 78)
print("EL MANUSCRITO CONTRA EL DEPOSITO")
print("=" * 78)

# ---------------------------------------------------------------- la escalera
lad = dat("results/ladder.dat")
i25 = lad["R"].index(2.50)
i11 = lad["R"].index(1.10)
i20 = lad["R"].index(2.00)

comprueba("Fig.10  3x: rango de la ganancia generativa",
          r"wins at \\emph\{every\} geometry \(\$\+([\d.]+)\$ to \$\+([\d.]+)\$~mHa",
          [min(lad["gap3"]), max(lad["gap3"])])

comprueba("Fig.10  1x: el punto de R=1.1 y su t",
          r"running from \$-([\d.]+)\\pm([\d.]+)\$~mHa at \$R\{=\}1\.1\$ .*?t_4\{=\}-([\d.]+)\$",
          [abs(lad["gap1"][i11]), lad["gap1sd"][i11], abs(lad["t1"][i11])])

comprueba("Fig.10  0x: el mas profundo, en R=2.5",
          r"deepest at \$R\{=\}2\.5\$, \$-([\d.]+)\$~mHa, \$t_4\{=\}-([\d.]+)\$",
          [abs(lad["gap0"][i25]), abs(lad["t0"][i25])])

comprueba("Fig.10  0x: el punto nulo de R=1.7",
          r"null \$\+([\d.]+)\\pm([\d.]+)\$~mHa at \$R\{=\}1\.7\$",
          [lad["gap0"][lad["R"].index(1.70)], lad["gap0sd"][lad["R"].index(1.70)]])

# ------------------------------------------------------- el parametro de orden
op = dat("results/orderparam.dat")
comprueba("Sec.7   rho declina de equilibrio a disociacion",
          r"falling from \$\\rho\{=\}([\d.]+)\$ near equilibrium.*?through \$([\d.]+)\$ at \$R=2\.0\$"
          r".*?to \$([\d.]+)\$ at \$R=2\.5\$",
          [round(op["spearman"][0], 2), round(op["spearman"][3], 2), round(op["spearman"][5], 2)],
          tol=0.006)

comprueba("Sec.7   el peso Hartree-Fock colapsa",
          r"Hartree--Fock weight \$([\d.]+)\$.*?HF weight \$([\d.]+)\$.*?occupations \$1\.08/0\.92\$",
          [round(1 - op["mrweight"][0], 2), round(1 - op["mrweight"][3], 2)], tol=0.006)

# --------------------------------------------------------- el barrido de suelo
s3 = jsn("results/fig6_suelo_1e-03.json")
s6 = jsn("results/fig6_suelo_1e-06.json")
s12 = jsn("results/fig6_suelo_1e-12.json")

comprueba("Sec.4.6 los tres suelos a D=120, contra el codicioso",
          r"the GFlowNet returns \$([\d.]+)\\pm([\d.]+)\$, \$([\d.]+)\\pm([\d.]+)\$, and --- at \$10\^\{-12\}\$"
          r" --- no value at all, against a deterministic greedy selector at \$([\d.]+)\$~mHa",
          [s3["stats"]["120"]["gflownet"]["mean"], s3["stats"]["120"]["gflownet"]["std"],
           s6["stats"]["120"]["gflownet"]["mean"], s6["stats"]["120"]["gflownet"]["std"],
           s3["det_greedy"]["120"]])

comprueba("Sec.4.6 el codicioso se aplana pasado D=91",
          r"\(\$([\d.]+)\$ at \$D\{=\}120\$, \$([\d.]+)\$ at \$D\{=\}240\$\)",
          [s3["det_greedy"]["120"], s3["det_greedy"]["240"]])

comprueba("Sec.4.5 la unica celda donde el proponente adelanta",
          r"\$D\{=\}60\$ at a floor of \$10\^\{-12\}\$, \$([\d.]+)\$ against \$([\d.]+)\$~mHa",
          [s12["stats"]["60"]["gflownet"]["mean"], s12["det_greedy"]["60"]])

if "120" in s12["stats"] and s12["stats"]["120"].get("gflownet"):
    fallos.append(("Sec.4.6 'no value at all' a 1e-12",
                   "el JSON SI trae un valor a D=120; la frase del paper es falsa"))
else:
    pasan += 1
    print("  OK  Sec.4.6 a 1e-12 el JSON efectivamente no llena D=120")

# ------------------------------------------------------- la lectura simetrica
sym_path = os.path.join(ROOT, "results", "lectura_simetrica.json")
if os.path.exists(sym_path):
    sym = jsn("results/lectura_simetrica.json")["summary"]
    comprueba("Sec.7   la lectura simetrica invierte el orden",
              r"degrading to \$([\d.]+)\\pm([\d.]+)\$~mHa against \$([\d.]+)\\pm([\d.]+)\$~mHa for the"
              r" classical recovery loop and \$([\d.]+)\\pm([\d.]+)\$~mHa raw",
              [sym["gflownet"]["mean"], sym["gflownet"]["std"],
               sym["ibm"]["mean"], sym["ibm"]["std"],
               sym["raw"]["mean"], sym["raw"]["std"]])
else:
    fallos.append(("Sec.7 lectura simetrica", "falta results/lectura_simetrica.json"))

# ---------------------------------------------------------------- la figura 7
# Es la figura que mas ha drifteado del deposito: sus dos barras clasicas llegaron a
# tener tres valores distintos (10.28 / 9.90 / 9.81) y sus dos barras generativas eran
# literales que ningun script producia. Ahora las cuatro salen de dos generadores.
_hci = os.path.join(ROOT, "results", "hci_baseline.json")
_gfn = os.path.join(ROOT, "results", "gfn_ruidoso_fig7.json")
if os.path.exists(_hci) and os.path.exists(_gfn):
    hb = jsn("results/hci_baseline.json")
    gb = jsn("results/gfn_ruidoso_fig7.json")["mols"]
    comprueba("Fig.7   las dos barras clasicas",
              r"coordinates \{\(0,([\d.]+)\) \(1,([\d.]+)\)\}; \\addlegendentry\{EN-selected CI",
              [hb["h2o"]["hci_mHa"], hb["n2"]["hci_mHa"]])
    comprueba("Fig.7   las dos barras generativas",
              r"coordinates \{\(0,([\d.]+)\) \(1,([\d.]+)\)\}; \\addlegendentry\{GFlowNet-SQD",
              [gb["h2o"]["gflownet"]["mean"], gb["n2"]["gflownet"]["mean"]])
    comprueba("Sec.5.5 el veredicto decisivo",
              r"reaches \$([\d.]+)\$~mHa on H\$_2\$O .*?and \$([\d.]+)\$~mHa on stretched N\$_2\$",
              [round(hb["h2o"]["hci_mHa"], 1), round(hb["n2"]["hci_mHa"], 1)])
else:
    fallos.append(("Fig.7 las cuatro barras",
                   "faltan results/hci_baseline.json o results/gfn_ruidoso_fig7.json"))


# ------------------------------------------------------------- el coleccionista
cw = dat("results/coupon_w.dat")
comprueba("Sec.3.1 el ancla invariante: el peso mayor",
          r"The largest single-string weight, \$([\d.]+)\$", [max(cw["weight"])], tol=1e-6)

# --------------------------------------------------------------- la recuperacion
rec = dat("results/recovery.dat")
if "lost" in rec:
    comprueba("Sec.3.2 fraccion de tiros perdidos a 3x",
              r"\(\$?\\sim\$?([\d]+)\\% lost at \$3\\times\$\)", [round(max(rec["lost"]))], tol=1.01)

print()
print("=" * 78)
if fallos:
    print("FALLAN %d COMPROBACIONES (%d pasan):" % (len(fallos), pasan))
    for etq, msg in fallos:
        print("  FALLA  %-52s %s" % (etq, msg))
    print()
    print("Una comprobacion que falla porque 'el patron no aparece' NO es un falso")
    print("positivo: significa que la frase del manuscrito cambio y nadie actualizo")
    print("este fichero, que es exactamente como los numeros se desincronizan.")
    sys.exit(1)
print("LAS %d COMPROBACIONES PASAN: el manuscrito y el deposito dicen lo mismo." % pasan)
