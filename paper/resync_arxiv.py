# -*- coding: utf-8 -*-
"""Rehace los paquetes de arXiv desde el paper y COMPILA EL ZIP EXTRAIDO.

Compilar el directorio de trabajo no prueba nada sobre el paquete: arXiv compila
lo que hay DENTRO del zip. Se extrae en un temporal limpio y se compila alli.

El manifiesto NO esta escrito a mano: se deriva de lo que main.tex realmente lee,
para que no se pueda quedar obsoleto cuando un fichero deja de usarse (paso con
wf_spec.dat y wf_subspace.dat, retirados por no leerse).
"""
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

PAPER = r"C:\Users\Nicolas\Downloads\Proyecto_SQD_ML\P1_review_ML_for_SQD\release\paper"
PDF_DEPOSITADO = os.path.join(PAPER, "main.pdf")
ZIPS = [
    os.path.join(PAPER, "arxiv-submission.zip"),
    r"C:\Users\Nicolas\Downloads\Proyecto_SQD_ML\P1_review_ML_for_SQD\work\arxiv_submission\arxiv_upload.zip",
]

tex = io.open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
leidos = {"main.tex"}
for m in re.finditer(r"table[^{]*\{([^}]+\.dat)\}", tex):
    leidos.add(os.path.basename(m.group(1)))
for m in re.finditer(r"\\includegraphics[^{]*\{([^}]+)\}", tex):
    b = os.path.basename(m.group(1))
    leidos.add(b if os.path.isfile(os.path.join(PAPER, b)) else b + ".pdf")

miembros = sorted(x for x in leidos if os.path.isfile(os.path.join(PAPER, x)))
falta = sorted(x for x in leidos if not os.path.isfile(os.path.join(PAPER, x)))
print("=" * 70)
print("manifiesto derivado de main.tex: %d ficheros" % len(miembros))
for x in miembros:
    print("   %s" % x)
if falta:
    print("FALTAN: %s" % falta)
    raise SystemExit(1)

tmpzip = os.path.join(tempfile.gettempdir(), "arxiv_nuevo.zip")
with zipfile.ZipFile(tmpzip, "w", zipfile.ZIP_DEFLATED) as z:
    for m in miembros:
        z.write(os.path.join(PAPER, m), m)
print("\npaquete: %.1f KB" % (os.path.getsize(tmpzip) / 1024))

d = tempfile.mkdtemp(prefix="arxiv_test_")
with zipfile.ZipFile(tmpzip) as z:
    z.extractall(d)
print("compilando la copia extraida ...")
for _ in range(3):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                   cwd=d, capture_output=True, timeout=400)
log = io.open(os.path.join(d, "main.log"), encoding="utf-8", errors="replace").read()
err = [l for l in log.split("\n") if l.startswith("!")]
pags = re.search(r"Output written.*?\((\d+) pages", log)
undef = "undefined" in log.lower()
over = log.count("Overfull")
print("  paginas: %s | errores: %d | undefined: %s | overfull: %d"
      % (pags.group(1) if pags else "?", len(err), undef, over))
if err:
    print("  PRIMER ERROR: %s" % err[0])

# el titulo, tal como saldra en el PDF del paquete
if "a review of generative configuration recovery" in tex:
    print("  titulo: contiene 'review' -> cumple el requisito de Quantum")
else:
    print("  AVISO: el titulo NO contiene 'review'")

ok = (not err) and (not undef) and over == 0 and pags
if not ok:
    print("\nEL ZIP NO COMPILA LIMPIO -- no se despliega.")
    raise SystemExit(1)

for z in ZIPS:
    os.makedirs(os.path.dirname(z), exist_ok=True)
    shutil.copy2(tmpzip, z)
    print("  actualizado %s" % z)

# EL PDF DEPOSITADO. Hasta el 2026-09-13 este script compilaba en un directorio
# temporal, copiaba el zip y tiraba el PDF -- asi que release/paper/main.pdf se
# quedaba en la ultima vez que alguien compilo a mano. Llego a tener un dia de
# retraso sobre main.tex, con el README enlazandolo. Es el mismo defecto que el
# paper documenta en otros ficheros, en el fichero que el lector abre primero.
pdf_compilado = os.path.join(d, "main.pdf")
if os.path.exists(pdf_compilado):
    shutil.copy2(pdf_compilado, PDF_DEPOSITADO)
    print("  actualizado %s  (el PDF que enlaza el README)" % PDF_DEPOSITADO)
else:
    print("  AVISO: no hay main.pdf compilado; el PDF depositado se queda obsoleto")
    raise SystemExit(1)
shutil.rmtree(d, ignore_errors=True)
print("\nLISTO: %s paginas, 0 errores, 0 overfull, compilado DESDE EL ZIP."
      % (pags.group(1) if pags else "?"))
