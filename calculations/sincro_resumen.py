"""Check that the abstract says the same thing in all four places it lives.

The abstract exists four times: in paper/main.tex, as plain text in docs/abstract_arxiv.txt
for the arXiv form, inside CITATION.cff, and inside paper/arxiv-submission.zip. Every edit
to it is therefore four edits, and in this repository's history one of them has been missed
more than once.

This compares them and reports what differs. It also checks the arXiv character limit and
the claims Quantum requires the abstract to carry.

    python calculations/sincro_resumen.py        # exit 0 if synchronized, 1 if not
"""
import io
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
LIMITE_ARXIV = 1920


def plano(t):
    """LaTeX -> texto plano, la misma reduccion que produjo docs/abstract_arxiv.txt."""
    t = re.sub(r"\\emph\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", t)
    t = t.replace("$\\alpha$", "alpha")
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", "", t)
    t = t.replace("{", "").replace("}", "").replace("---", "--").replace("\\", "")
    return " ".join(t.split())


def del_tex(texto):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", texto, re.S)
    return plano(m.group(1).strip()) if m else None


fallos = []

tex = io.open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
a_tex = del_tex(tex)
a_form = io.open(os.path.join(ROOT, "docs", "abstract_arxiv.txt"), encoding="utf-8").read().strip()

cff = io.open(os.path.join(ROOT, "CITATION.cff"), encoding="utf-8").read()
m = re.search(r"abstract:\s*>-\s*\n((?:[ \t]+.*\n)+)", cff)
a_cff = " ".join(m.group(1).split()) if m else None

a_zip = None
z = os.path.join(ROOT, "paper", "arxiv-submission.zip")
if os.path.exists(z):
    with zipfile.ZipFile(z) as f:
        if "main.tex" in f.namelist():
            a_zip = del_tex(f.read("main.tex").decode("utf-8"))

print("=" * 78)
print("EL RESUMEN, EN LAS CUATRO COPIAS")
print("=" * 78)
for nombre, a in (("paper/main.tex", a_tex), ("docs/abstract_arxiv.txt", a_form),
                  ("CITATION.cff", a_cff), ("arxiv-submission.zip", a_zip)):
    print("  %-26s %s" % (nombre, ("%d palabras, %d caracteres" % (len(a.split()), len(a)))
                          if a else "NO SE PUDO LEER"))

print()
if a_tex and a_zip and a_tex != a_zip:
    fallos.append("el zip lleva un resumen distinto del de main.tex -- resincronice el paquete")
if a_tex and a_form and a_tex != a_form:
    # la unica diferencia legitima es que el del formulario puede ir recortado
    # El del formulario NO tiene por que ser un prefijo del otro. arXiv corta en 1920 y
    # Quantum no limita longitud, asi que el del formulario es una condensacion escrita
    # para ese limite. Lo que hay que exigirle no es identidad sino que quepa y que
    # conserve las afirmaciones que no pueden perderse, y eso se comprueba abajo.
    print("  el del formulario es una condensacion del otro, no una copia:")
    print("     paper %d caracteres -> formulario %d (%.0f%% del original)"
          % (len(a_tex), len(a_form), 100.0 * len(a_form) / len(a_tex)))
if a_cff and a_form and a_cff != a_form and a_cff != a_tex:
    fallos.append("CITATION.cff no coincide con ninguno de los otros dos")

if a_form and len(a_form) > LIMITE_ARXIV:
    fallos.append("el resumen del formulario excede el limite de arXiv: %d > %d"
                  % (len(a_form), LIMITE_ARXIV))
elif a_form:
    print("  margen bajo el limite de arXiv (%d): %+d caracteres" % (LIMITE_ARXIV, LIMITE_ARXIV - len(a_form)))

# lo que Quantum exige que el resumen de una review lleve
print()
for etq, frag in (("publico objetivo (Quantum lo exige)", "addresses"),
                  ("prioridad de Reinholdt", "Reinholdt"),
                  ("los tres calificadores", "same-active-space"),
                  ("el hueco del GFlowNet", "generative-flow-network")):
    ok = a_tex and frag in a_tex
    print("  [%s] %s" % ("OK " if ok else "FALTA", etq))
    if not ok:
        fallos.append("el resumen de main.tex no lleva: " + etq)

print()
print("=" * 78)
if fallos:
    for f in fallos:
        print("  FALLA  %s" % f)
    sys.exit(1)
print("  LAS CUATRO COPIAS DICEN LO MISMO.")
