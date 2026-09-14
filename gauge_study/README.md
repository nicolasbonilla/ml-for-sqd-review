# Estudio de gauge — qué produce cada script

Todo esto corre en Docker con PySCF; **PySCF no publica ruedas para Windows**, así que
en Windows es Docker o WSL, no hay tercera opción.

```bash
printf 'FROM python:3.11-slim\nRUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*\nRUN pip install --no-cache-dir numpy scipy pyscf\n' > Dockerfile
docker build -t sqd-fci .
docker run --rm -v "$PWD":/w -w /w sqd-fci python /w/<script>.py
```

**Fije los hilos** (`-e OMP_NUM_THREADS=1`) si quiere reproducir orientaciones orbitales
bit a bit: véase `determinismo.py`.

| script | qué establece |
|---|---|
| `fci_forense.py` | La receta declarada reproduce `E_FCI = −108,808041914843 Ha`. Identifica las tres capas π degeneradas del espacio activo y mide que rotar dentro de ellas deja la energía intacta y mueve los pesos. |
| `simetria.py` | Las tres capas son **E1** en D∞h — es decir π, no δ (que sería E2). Cuatro capas darían 40 cadenas invariantes, no 84, así que "tres" está forzado. |
| `verificacion_final.py` | Con `scf.conv_tol=1e-12` y `fci.conv_tol=1e-13`, w(11) y w(12) son **exactamente degenerados**. Completando el multiplete, el conteo del 90 % da **12 en las 40 rotaciones**. El del 99,9 % **no** es robusto: 185–196. |
| `invariancia_general.py` | La prueba fuerte: el espectro de Schmidt α\|β es invariante bajo **cualquier** rotación orbital de espín colineal — 8 rotaciones generales de SO(12), orbitales naturales y de Boys dan **9/21/60** con diferencias de 10⁻¹⁵, mientras el conteo de cadenas se mueve de 10 a 450. |
| `cota_schmidt.py` | La cota `\|S\| ≥ r_Schmidt(p)²` sobre subespacios producto: **0 violaciones en 4.000 subespacios aleatorios**. Para N₂ al 90 %: suelo de 81 determinantes en cualquier base. No es ajustada — el mejor producto top-k necesita 169. |
| `c2_multiplete.py` | El contraejemplo de § 3.1: en **C₂ a 1,24 Å** completar el multiplete **ensancha** el rango del 90 %, de 4–5 a **4–6**. Es decir, que la estabilización que se observa en N₂ (donde completar el multiplete fija el conteo en 12) **no es una regla general**, y el paper lo dice así. Usa `fci.addons.transform_ci_for_orbital_rotation`: volver a correr CASCI con los orbitales rotados falla, porque PySCF adaptado por simetría los rechaza. |
| `rho_sensibilidad.py` | Mide cuanto mueven a rho el umbral y el gauge, por separado. Sobre 41 gauges (40 rotaciones intra-capa mas el adaptado por simetria) y con E_FCI invariante a 2,3·10⁻¹³ Ha: sin umbral rho sube 0,085 de media y su dispersion de gauge es 0,063; con umbral, 0,034. Comprueba ademas que el espectro de recompensa es bimodal — la senal empieza en 1,2·10⁻¹² del maximo y el residuo no pasa de ~10⁻³⁴ — que es lo que hace que el corte no sea un boton. Refuto dos cifras del manuscrito, que se corrigieron. |
| `bloque_conteos.py` | El bloque de conteos de § 3.1: las 84 de 792 cadenas invariantes (10,6 %), las **6 / 23 / 82 orbitas (13 / 54 / 208 cadenas)** con rango **cero** sobre 40 rotaciones, los conteos crudos oscilando 11–12, 46–50 y 185–196, y el cociente peor-caso (1,09 en N₂, 1,25 en C₂). Se niega a comparar con el sorteo canonico si la energia no coincide: en C₂ el CASCI canonico converge a **otro estado**, 14,8 mHa mas abajo, y comparar conteos entre estados distintos no dice nada del gauge. |
| `determinismo.py` | **Los orbitales canónicos RHF no son reproducibles** con capas degeneradas: con hilos por defecto, cuatro corridas dan orientaciones π de 30,05° / 13,83° / 64,94° / 144,92°. Con `OMP_NUM_THREADS=1`, las cuatro dan 130,192°. Con `mol.symmetry=True`, 90,000° siempre. |

## La trampa que costó más cara — y lo que el barrido corrigió

`np.maximum(w, 1e-12)` frente a `1e-3 * w.max()` en el suelo de la recompensa del
GFlowNet son nueve órdenes de magnitud y mueven el error de la Fig. 6 en más de un factor
dos. **No cambian quién gana.**

La v2 barrió el suelo en vez de elegirlo — tres valores × cinco semillas, todo lo demás
fijo (`../figures/barrido_suelo.py`). A D = 120 el GFlowNet da 110,8 ± 11,0 mHa (suelo
1e-3), 58,2 ± 6,6 (1e-6) y **ningún valor** a 1e-12, contra 46,4 mHa del selector
codicioso determinista. Bajarlo mejora al proponente de forma monótona pero nunca lo
adelanta a esa dimensión, y por debajo de cierto punto deja de poder *llenar* el
subespacio: sólo **91 de las 792 cadenas** llevan recompensa barata no nula
(`../figures/cuenta_rankeables.py`). En todo el barrido el proponente adelanta en
**exactamente una celda** — D = 60 a 1e-12, 51,6 contra 54,6 mHa, sin resolver a cinco
semillas.

> ⚠️ Una versión anterior de este fichero decía que el suelo «le da la vuelta a la
> conclusión». Se escribió desde un solo ajuste, antes de que el barrido existiera. Es
> falso. La § 4.6 del paper lleva la versión corregida.

El script que sí reproduce la Fig. 6 es `../figures/fig6_5seed.py`; `compact_fig.py` de la
v1 no, y se conserva sólo como `../figures/OBSOLETO_compact_fig_v1.py`.

Y una de formato: los `.dat` comentan su cabecera con `%`, no con `#`, porque
**pgfplots no trata `#` como comentario** — lee la cabecera como datos y destroza la
gráfica sin dar ni un error de compilación.

Pero la fila de **nombres de columna** no está comentada, porque pgfplots la necesita —
así que `np.loadtxt(f, comments='%')` **falla en todos los ficheros** de este
repositorio. Las dos recetas que sí funcionan están en el README principal, sección
*Two traps that will cost you a day*; ambas se ejecutan tal cual y están probadas
contra los quince `.dat`.
