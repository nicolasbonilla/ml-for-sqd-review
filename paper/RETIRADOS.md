# Ficheros retirados del deposito el 2026-09-13

`paper/build_notebook.py` construia la version v1 del notebook. Reconstruia, entre
otras cosas, el parametro de orden como 0,72 -> 0,60 y la comparacion de compacidad
como 192 +- 19 frente a 41,3 mHa -- todas cifras que la v2 retracta -- y llevaba el
titulo anterior. Mientras estuviese en el repositorio, cualquiera podia regenerar el
notebook y reinyectar lo retractado sin darse cuenta. El notebook
`notebook/GFlowNet_SQD_calculations.ipynb` se mantiene ahora directamente y se
reejecuta entero; sus celdas de produccion leen los ficheros de `results/`, asi que ya
no hay un segundo sitio donde esos numeros puedan quedarse obsoletos.

`paper/paper_build.py` leia `Paper_Review_ML_SQD_borrador.md`, que nunca estuvo en el
deposito, y escribia `paper.tex`, que no es el manuscrito. Era un resto de un pipeline
de borrador y no construia nada de lo que aqui se publica.

Ambos siguen en el historial de git si hicieran falta.

`calculations/rho_repair_ladder_json.py` existio unas horas el 2026-09-13. Reparaba la
columna rho de `results/n2_ladder_crossover.json`, que era una corrida SUPERADA de dos
escalas de ruido cuyos gaps de energia contradecian a `ladder.dat` -- en R=2,00 daba
-6,91 mHa donde el .dat dice -0,72, y en R=1,70 hasta el signo cambiaba. Reparar la rho
de ese fichero no arreglaba el problema de fondo: lo dejaba con apariencia de vigente.

Se reejecuto entero `calculations/n2_ladder_crossover.py`. El `.dat` que produjo
coincide con el depositado **en las 32 columnas, con diferencia maxima 0,00** -- o sea
que `ladder.dat` SI es regenerable desde su generador declarado, comprobado por primera
vez. El JSON de esa misma corrida sustituye al superado, asi que los dos ficheros salen
ahora de una sola ejecucion y no pueden volver a contradecirse. El script de reparacion
se queda sin objeto.
