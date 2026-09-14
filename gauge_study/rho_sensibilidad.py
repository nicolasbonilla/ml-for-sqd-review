# -*- coding: utf-8 -*-
r"""Mide las tres cifras que la seccion 7 afirma sobre rho y que no tenian generador.

HISTORIA. La seccion 7 afirmaba: "Without it, rho is inflated by ~0.2 and acquires a
spurious gauge sensitivity of 0.18; with it, the residual gauge spread is 0.03", sin
generador que lo respaldase. Este fichero los midio, dos de los tres no se sostuvieron
y el manuscrito se corrigio. Lo que el paper dice HOY es lo que se contrasta abajo:
inflacion 0.085, dispersion sin umbral 0.063, dispersion con umbral 0.034.

QUE SE MIDE. Con N2 fijo a R=2.0 A en CAS(10e,12o), se generan NROT rotaciones
aleatorias DENTRO de las capas pi degeneradas. Cada rotacion deja E_FCI intacta y
redistribuye el peso entre cadenas alpha, asi que cada una es un gauge legitimo. Para
cada gauge se calcula rho de las dos maneras:

  rho_cruda        Spearman(recompensa Epstein-Nesbet, |c|^2 exacto) sin umbral
  rho_umbralizada  lo mismo con las entradas < 1e-12*max puestas a cero antes de rankear

y se reportan (i) la dispersion de cada una sobre los gauges, que es la sensibilidad de
gauge, y (ii) la diferencia rho_cruda - rho_umbralizada, que es la inflacion.

POR QUE HAY QUE ROTAR LOS INTEGRALES Y NO SOLO EL VECTOR CI. La recompensa se construye
desde los integrales del espacio activo, asi que vive en la misma base que el vector. Se
rotan los dos: h1' = U^T h1 U, h2' la transformacion de cuatro indices, y el vector con
fci.addons.transform_ci_for_orbital_rotation. La comprobacion de que las dos
transformaciones usan el MISMO convenio es que <C'|H'|C'> debe devolver E_FCI; el script
la hace y se niega a reportar nada si falla.

Correr con OMP_NUM_THREADS=1:
    docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci \
        python /w/gauge_study/rho_sensibilidad.py
"""
import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.fci import direct_spin1
from scipy.stats import spearmanr

NCAS, NELECAS = 12, (5, 5)
na, nb = NELECAS
R, NROT, SEMILLA = 2.0, 40, 0
RHO_FLOOR = 1e-12
TOL_E = 1e-9          # Ha: la rotacion debe dejar la energia intacta a este nivel


def recompensa_y_pesos(h1, h2, C):
    """Recompensa Epstein-Nesbet marginalizada sobre beta, y |c|^2 exacto, por cadena."""
    dim = C.shape[0]
    hf = (1 << na) - 1
    from pyscf.fci import cistring
    strs = cistring.make_strings(range(NCAS), na)
    hf_idx = int(np.where(strs == hf)[0][0])

    v0 = np.zeros((dim, dim))
    v0[hf_idx, hf_idx] = 1.0
    h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
    Hc = direct_spin1.contract_2e(h2e, v0, NCAS, NELECAS).reshape(dim, dim)
    hd = direct_spin1.make_hdiag(h1, h2, NCAS, NELECAS).reshape(dim, dim)
    den = Hc[hf_idx, hf_idx] - hd
    den[hf_idx, hf_idx] = 1.0
    c1 = Hc / den
    c1[hf_idx, hf_idx] = 1.0
    w_cheap = (c1 ** 2).sum(1)
    w_cheap /= w_cheap.sum()

    w_true = (C ** 2).sum(1)
    w_true /= w_true.sum()
    return w_cheap, w_true


def rhos(w_cheap, w_true):
    w_thr = np.where(w_cheap < RHO_FLOOR * w_cheap.max(), 0.0, w_cheap)
    return (float(spearmanr(w_cheap, w_true).correlation),
            float(spearmanr(w_thr, w_true).correlation),
            int((w_thr > 0).sum()))


def hueco(w_cheap):
    """El hueco vacio del espectro de recompensa que hace que el corte no sea un boton.

    La seccion 7 afirma que el espectro es BIMODAL: la senal empieza en 1.2e-12 del
    maximo y el residuo no pasa de ~1e-34, y que solo en el peor gauge llegan a caer 2 de
    las 792 entradas entre 1e-16 y 1e-12. (Una version anterior afirmaba que esa banda
    estaba vacia en todo gauge; esta medida la refuto y el manuscrito se corrigio.) Se
    devuelve cuantas caen dentro y los bordes reales del hueco."""
    r = np.sort(w_cheap / w_cheap.max())
    r = r[r > 0]
    dentro = int(((r > 1e-16) & (r < 1e-12)).sum())
    abajo = r[r <= 1e-16]
    arriba = r[r >= 1e-12]
    return dentro, (float(abajo.max()) if abajo.size else 0.0), float(arriba.min())


def energia(h1, h2, C, ecore):
    h2e = direct_spin1.absorb_h1e(h1, h2, NCAS, NELECAS, 0.5)
    Hc = direct_spin1.contract_2e(h2e, C, NCAS, NELECAS)
    return float(np.vdot(C.ravel(), Hc.ravel()).real) + ecore


def main():
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz", symmetry=True, verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.run()
    cas = mcscf.CASCI(mf, NCAS, NELECAS); cas.fcisolver.conv_tol = 1e-13
    e0 = cas.kernel()[0]
    C0 = np.asarray(cas.ci)
    h1, ecore = cas.get_h1cas()
    h2 = ao2mo.restore(1, cas.get_h2cas(), NCAS)

    eps = mf.mo_energy[cas.ncore:cas.ncore + NCAS]
    capas, i = [], 0
    while i < NCAS:
        j = i
        while j + 1 < NCAS and abs(eps[j + 1] - eps[i]) < 1e-8:
            j += 1
        if j > i:
            capas.append(list(range(i, j + 1)))
        i = j + 1

    print(f"N2 R={R} A  CAS(10e,12o)  cc-pVDZ  gauge de partida: adaptado por simetria")
    print(f"E_FCI = {e0:.12f} Ha")
    print(f"capas degeneradas del espacio activo: {capas}")
    ocupadas = [c for c in capas if all(o < na for o in c)]
    print(f"  de las cuales enteramente ocupadas en HF: {ocupadas}  "
          f"(ninguna capa cruza el borde de HF, luego el determinante HF es invariante)")

    r_raw0, r_thr0, nrank0 = rhos(*recompensa_y_pesos(h1, h2, C0))
    print(f"\ngauge adaptado por simetria:  rho_cruda={r_raw0:.4f}  "
          f"rho_umbralizada={r_thr0:.4f}  cadenas rankeables={nrank0}")

    rng = np.random.default_rng(SEMILLA)
    raw, thr, dE, huecos = [], [], [], [hueco(recompensa_y_pesos(h1, h2, C0)[0])]
    for k in range(NROT):
        U = np.eye(NCAS)
        for capa in capas:
            d = len(capa)
            Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
            U[np.ix_(capa, capa)] = Q
        h1r = U.T @ h1 @ U
        h2r = np.einsum("pqrs,pi,qj,rk,sl->ijkl", h2, U, U, U, U, optimize=True)
        Cr = np.asarray(fci.addons.transform_ci_for_orbital_rotation(C0, NCAS, NELECAS, U))
        e = energia(h1r, h2r, Cr, ecore)
        dE.append(abs(e - e0))
        wc_r, wt_r = recompensa_y_pesos(h1r, h2r, Cr)
        a, b, _ = rhos(wc_r, wt_r)
        raw.append(a); thr.append(b); huecos.append(hueco(wc_r))

    dE = np.array(dE)
    if dE.max() > TOL_E:
        print(f"\nNO SE REPORTA NADA: la rotacion movio la energia hasta {dE.max():.2e} Ha.")
        print("El convenio de transform_ci_for_orbital_rotation y el de los integrales no")
        print("coinciden; pruebe U.T. Sin esta comprobacion las cifras no significan nada.")
        raise SystemExit(1)

    # El gauge de partida es un gauge legitimo mas -- y es ADEMAS aquel en el que el
    # paper cita rho, asi que dejarlo fuera de la familia subestimaria la dispersion.
    # Resulta estar en el borde de la distribucion, no en su centro.
    raw = np.array(raw + [r_raw0]); thr = np.array(thr + [r_thr0])
    infl = raw - thr
    print(f"\ncomprobacion: E invariante bajo las {NROT} rotaciones a {dE.max():.2e} Ha")
    print("=" * 74)
    print(f"{'':>22s} {'min':>9s} {'max':>9s} {'dispersion':>12s} {'media':>9s}")
    for nom, v in (("rho SIN umbral", raw), ("rho CON umbral", thr),
                   ("inflacion (sin-con)", infl)):
        print(f"{nom:>22s} {v.min():9.4f} {v.max():9.4f} "
              f"{v.max()-v.min():12.4f} {v.mean():9.4f}")

    print("=" * 74)
    print("CONTRA LO QUE AFIRMA LA SECCION 7 DEL PAPER:")
    for etiqueta, medido, afirmado in (
            ("sensibilidad de gauge SIN umbral", raw.max() - raw.min(), 0.063),
            ("dispersion residual CON umbral", thr.max() - thr.min(), 0.034),
            ("inflacion por no umbralizar", infl.mean(), 0.085)):
        ok = abs(medido - afirmado) <= 0.15 * afirmado
        print(f"  {etiqueta:<36s} afirma {afirmado:5.2f}   mide {medido:6.4f}   "
              f"{'respaldado' if ok else '<-- NO RESPALDADO'}")
    print("=" * 74)
    print("EL HUECO ESPECTRAL QUE HACE QUE EL CORTE NO SEA UN BOTON")
    dentro = max(h[0] for h in huecos)
    print("  entradas entre 1e-16 y 1e-12 del maximo, en el PEOR de los %d gauges: %d"
          % (len(huecos), dentro))
    print("  borde inferior del hueco (max entrada <= 1e-16): %.2e * max"
          % max(h[1] for h in huecos))
    print("  borde superior del hueco (min entrada >= 1e-12): %.2e * max"
          % min(h[2] for h in huecos))
    print("  => el corte en 1e-12 %s"
          % ("cae en region VACIA en todo gauge: no es ajustable"
             if dentro == 0 else "NO esta en region vacia: LA AFIRMACION FALLA"))
    print()
    print("Nota de lectura: la inflacion es una cantidad POR GAUGE. Comparar la rho cruda")
    print("de un gauge cualquiera con la umbralizada de OTRO mezcla los dos efectos y da")
    print("una cifra mayor que cualquiera de los dos por separado.")


if __name__ == "__main__":
    main()
