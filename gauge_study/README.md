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
| `determinismo.py` | **Los orbitales canónicos RHF no son reproducibles** con capas degeneradas: con hilos por defecto, cuatro corridas dan orientaciones π de 30,05° / 13,83° / 64,94° / 144,92°. Con `OMP_NUM_THREADS=1`, las cuatro dan 130,192°. Con `mol.symmetry=True`, 90,000° siempre. |

## La trampa que costó más cara

`np.maximum(w, 1e-12)` frente a `1e-3 * w.max()` en el suelo de la recompensa del
GFlowNet cambia el resultado de la Fig. 6 en un factor diez y **le da la vuelta a la
conclusión**. Está documentado en `../figuras_scripts/fig6_5seed.py`.

Y una de formato: los `.dat` comentan su cabecera con `%`, no con `#`, porque
**pgfplots no trata `#` como comentario** — lee la cabecera como datos y destroza la
gráfica sin dar ni un error de compilación. Desde numpy: `comments='%'`.
