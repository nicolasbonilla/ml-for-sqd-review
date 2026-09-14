# Lo que hay aqui, y por que no debe leerse como vigente

`v1_2026-08-05_arXiv-2608.05314/` es la **primera version publica, congelada tal cual se
deposito en arXiv**. Se conserva a proposito: un lector que cite la v1 tiene que poder ver
exactamente lo que cito.

**Sus numeros estan superados, y varios estan retractados.** En particular:

| en la v1 congelada | en la version actual |
|---|---|
| `orderparam.dat` da rho = 0,7245 -> 0,5982 | 0,4723 -> 0,3790 |
| el conteo de determinantes se da sin declarar el gauge orbital | el gauge se declara (D-infinito-h, adaptado por simetria) y se cuantifica su efecto |
| se afirma que el teorema de Brillouin anula las cadenas alpha simplemente excitadas | no las anula: la recompensa marginaliza sobre beta, y esas 35 cadenas llevan el 66,6 % de la masa |
| `wf_occ.dat` y `wf_hist.dat` parten un par pi degenerado | regenerados desde la receta declarada, con el par identico |

La rho de la v1 se calculaba sin declarar umbral, asi que el coeficiente de rangos ordenaba
residuo de coma flotante (hasta 1e-42) por encima de ceros algebraicos. La seccion 7 del
manuscrito actual explica el defecto y lo corrige.

**No use ningun numero de este directorio.** Los vigentes estan en `../results/` y
`../paper/`, y `../calculations/verifica_deposito.py` comprueba que coinciden con lo que el
manuscrito imprime.
