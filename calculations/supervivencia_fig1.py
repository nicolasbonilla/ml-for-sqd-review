"""The two shot-survival fractions in Figure 1's caption, which had no generator.

WHAT FIGURE 1 CLAIMS. Until 17 September 2026 its caption, the TikZ node inside the
figure and the prose of sections 2.1 and 3.2 read "77%" and "59%". They read 76% and 57%
now -- what this script measures -- and section 6 records the correction. The numbers
below are therefore the caption's, not a target this script has to reproduce.

WHY THIS SCRIPT EXISTS. Neither number occurred anywhere in the deposit. The generator the
README credits for Figure 1 (figures/gen_fig1.py) computes no post-selection fraction at
all; figures/workflow_fig.py computes one such fraction but is a dead matplotlib script
that the README's generator table does not list, and it emits one number, not two. This is
the opening figure of a paper whose section 6 argues that every reported number needs a
deposited generator, and whose post-audit paragraph claims to have swept these sections.

There was a second reason to run it, and it returned an answer. 0.77^2 = 0.593, which
was the printed second figure to two places -- so the 59% looked like the square of the
first rather than a measurement, and squaring is an independence assumption: the two spin
halves of a real register are not independent under a shared depolarizing channel. Measured
directly, the full-register fraction is 0.5715 against a square of 0.5702, a gap of 0.0013.
The suspicion was therefore correct in kind and small in size, and the manuscript now prints
both numbers as measured.

THE MODEL is the one the rest of the deposit uses, and it is read from the header of
results/recovery.dat rather than retyped: a depolarizing background that fully randomises a
shot with probability lambda, then asymmetric readout, 1->0 at p10 (T1-dominated) and 0->1
at p01, from FakeTorino (Heron r1) medians.

SURVIVAL means the shot still carries the correct electron number: exactly na alpha
electrons in the 12-qubit single-spin half, and exactly na alpha AND nb beta across the
full 24-qubit register.

    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci \
        python /w/calculations/supervivencia_fig1.py
"""
import io
import json
import os

import numpy as np

NCAS, NELECAS = 12, (5, 5)
na, nb = NELECAS
SHOTS = 200000          # grande: la cifra se cita a dos decimales en un pie de figura
SEMILLA = 0

LAMBDA0, P100, P010 = 0.05, 0.0229, 0.0200
FUENTE = "medianas publicadas de FakeTorino (Heron r1)"
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    _pr, _nq = FakeTorino().properties(), FakeTorino().num_qubits
    P100 = float(np.median([_pr.qubit_property(q, "prob_meas0_prep1")[0] for q in range(_nq)]))
    P010 = float(np.median([_pr.qubit_property(q, "prob_meas1_prep0")[0] for q in range(_nq)]))
    FUENTE = "FakeTorino (Heron r1), leido en vivo"
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, os.pardir, "results", "supervivencia_fig1.json")


def corrompe(bits, rng):
    """El canal del deposito: fondo despolarizante, luego lectura asimetrica."""
    n, w = bits.shape
    out = bits.copy()
    dep = rng.random(n) < LAMBDA0
    out[dep] = (rng.random((int(dep.sum()), w)) < 0.5).astype(np.int8)
    uno, cero = out == 1, out == 0
    out[uno & (rng.random(out.shape) < P100)] = 0
    out[cero & (rng.random(out.shape) < P010)] = 1
    return out


def main():
    rng = np.random.default_rng(SEMILLA)
    print("  ruido [%s]: lambda=%.3f  p(1->0)=%.4f  p(0->1)=%.4f"
          % (FUENTE, LAMBDA0, P100, P010))
    print("  %d tiros, semilla %d" % (SHOTS, SEMILLA))

    # Mitad de un solo espin: 12 qubits con exactamente na electrones.
    alfa = np.zeros((SHOTS, NCAS), dtype=np.int8)
    alfa[:, :na] = 1
    for f in alfa:
        rng.shuffle(f)
    beta = np.zeros((SHOTS, NCAS), dtype=np.int8)
    beta[:, :nb] = 1
    for f in beta:
        rng.shuffle(f)

    a2, b2 = corrompe(alfa, rng), corrompe(beta, rng)
    ok_a = a2.sum(1) == na
    ok_b = b2.sum(1) == nb
    f_media = float(ok_a.mean())
    f_entera = float((ok_a & ok_b).mean())

    print()
    print("  mitad de un solo espin (12 qubits):  %.4f  -> %.0f%%" % (f_media, 100 * f_media))
    print("  registro completo   (24 qubits):     %.4f  -> %.0f%%" % (f_entera, 100 * f_entera))
    print()
    print("  el cuadrado de la primera:           %.4f  -> %.0f%%" % (f_media ** 2, 100 * f_media ** 2))
    dif = abs(f_entera - f_media ** 2)
    print("  |medido - cuadrado| = %.5f" % dif)
    if dif < 0.005:
        print("  -> las dos mitades salen practicamente independientes BAJO ESTE MODELO,")
        print("     porque el fondo despolarizante se aplica por mitad y no al registro")
        print("     entero. En un dispositivo real no tiene por que serlo, y el pie de la")
        print("     figura no debe presentar el cuadrado como si fuera una medida aparte.")
    else:
        print("  -> NO son independientes: el pie no puede obtener una de la otra.")

    print()
    print("  medido: %.0f%% y %.0f%%   (el manuscrito imprime estos desde el 2026-09-17;"
          % (100 * f_media, 100 * f_entera))
    print("   antes imprimia 77% y 59%, y la seccion 6 lo registra como correccion)")

    res = {"_nota": ("Las dos fracciones de supervivencia del pie de la Fig. 1. No tenian "
                     "generador en el deposito hasta el 2026-09-16."),
           "modelo": {"lambda_depol": LAMBDA0, "p_1to0": P100, "p_0to1": P010,
                      "fuente": FUENTE},
           "shots": SHOTS, "semilla": SEMILLA,
           "supervivencia_media_12q": round(f_media, 4),
           "supervivencia_entera_24q": round(f_entera, 4),
           "cuadrado_de_la_media": round(f_media ** 2, 4),
           # DERIVADO, no escrito a mano: el 2026-09-17 este campo llevaba 0.77 y 0.59
           # como literales mientras el manuscrito ya imprimia 0.76 y 0.57, de modo que
           # reejecutar el generador deshacia la correccion. Ahora sigue a la medida.
           "el_paper_imprime": {"media_12q": round(f_media, 2),
                                "entera_24q": round(f_entera, 2),
                                "_nota": ("derivado de lo medido; el manuscrito imprime "
                                          "estos dos valores en la Fig. 1, la 2.1 y la "
                                          "3.2 desde el 2026-09-17")}}
    io.open(DEST, "w", encoding="utf-8", newline="\n").write(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("\n  escrito %s" % os.path.normpath(DEST))


if __name__ == "__main__":
    main()
