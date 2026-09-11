<div align="center">

# Machine learning for sample-based quantum diagonalization
### *generative configuration recovery and the classical-simulability frontier*

**Nicolás Bonilla Vargas** &nbsp;[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--6155--4391-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0006-6155-4391)

[![arXiv](https://img.shields.io/badge/arXiv-2608.05314-b31b1b.svg)](https://arxiv.org/abs/2608.05314)
[![Paper](https://img.shields.io/badge/paper-PDF%20(36%20pp)-blue.svg)](paper/main.pdf)
[![Type](https://img.shields.io/badge/type-review%20%2F%20perspective-8A2BE2.svg)](paper/main.pdf)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Text: CC BY 4.0](https://img.shields.io/badge/text-CC--BY--4.0-lightgrey.svg)](LICENSE)

</div>

---

A critical review of machine learning for sample-based quantum diagonalization (SQD),
equivalently quantum-selected configuration interaction (QSCI). It organizes the
ecosystem of generative and learned configuration selectors, reports a carefully scoped
negative on whether the quantum sampler beats classical selected CI, distils a
benchmarking standard, and tests that standard on FCI-exact systems — **including
against the paper's own positive result.**

**The manuscript is `paper/main.tex`. It is the only live manuscript in this
repository.**

---

## Every figure, and what produces it

The paper has ten figures. **Two are schematics that live entirely in the LaTeX source
and have no data behind them** — that is stated here rather than left for you to
discover. The other eight are listed with the script that produces their data.

| Figure | Source | Produced by |
|---|---|---|
| 1 `fig:loop` — SQD workflow | `wf_occ.dat`, `wf_hist.dat` | [`figures/gen_fig1.py`](figures/gen_fig1.py) |
| 2 `fig:system` — molecular orbitals | `pf_system.pdf` | [`figures/compose_fig2.py`](figures/compose_fig2.py) + `gen_cubes.py` + `render_orb.py` (PyMOL) |
| 3 `fig:coupon` — coupon-collector | `coupon_w.dat`, `coupon_cum.dat` | [`figures/gen_coupon.py`](figures/gen_coupon.py) |
| 4 `fig:score` — S-CORE recovery | `recovery.dat` | [`figures/data_figs.py`](figures/data_figs.py) (§ recovery panel) |
| 5 `fig:taxonomy` — method taxonomy | — | **schematic; pure TikZ in `main.tex`** |
| 6 `fig:compact` — GFlowNet compactness | numbers inline in `main.tex` | [`figures/fig6_5seed.py`](figures/fig6_5seed.py) ⚠️ read the trap below |
| 7 `fig:hci` — classical selected CI | numbers inline in `main.tex` | [`calculations/hci_baseline.py`](calculations/hci_baseline.py) |
| 8 `fig:landscape` — regime map | — | **schematic; pure TikZ in `main.tex`** |
| 9 `fig:orderparam` — order parameter | `orderparam.dat` | [`calculations/compute_physics.py`](calculations/compute_physics.py) |
| 10 `fig:crossover` — noise ladder | `ladder.dat` | [`calculations/n2_ladder_crossover.py`](calculations/n2_ladder_crossover.py) |

All ten figures are **native TikZ/PGFPlots inside `main.tex`** — there are no raster
figures and no external figure PDFs except the ray-traced orbital panel of Fig. 2.
The scripts above produce the *data*; the drawing lives in the manuscript.

## Two traps that will cost you a day

**1 · The reward floor of Fig. 6.** `np.maximum(w, 1e-12)` versus
`FLOOR = 1e-3 * w.max()` — nine orders of magnitude — changes the result by a factor of
ten **and reverses the conclusion.** The v1 script `compact_fig.py` used the former and
does *not* reproduce Fig. 6; it is kept only as
[`figures/OBSOLETO_compact_fig_v1.py`](figures/OBSOLETO_compact_fig_v1.py) with a
warning header. **Use `fig6_5seed.py`.**

**2 · `%` is a comment, `#` is not.** pgfplots does **not** treat `#` as a comment
character: a `#`-commented header is read as data and silently destroys the plot with no
compilation error. Every `.dat` here comments its header with `%`. From numpy, read them
with `comments='%'`.

## Reproducing

PySCF publishes no Windows wheels, so on Windows it is Docker or WSL:

```bash
printf 'FROM python:3.11-slim\nRUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*\nRUN pip install --no-cache-dir numpy scipy pyscf\n' > Dockerfile
docker build -t sqd-fci .
docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci python /w/figures/gen_fig1.py
```

**Pin the threads.** With degenerate shells, canonical RHF orbitals are *not*
reproducible run to run: four runs with default threading gave π orientations of
30.05° / 13.83° / 64.94° / 144.92°; with `OMP_NUM_THREADS=1`, all four gave 130.192°.
See [`gauge_study/determinismo.py`](gauge_study/determinismo.py).

The system throughout is **N₂, R = 2.0 Å, cc-pVDZ, CAS(10e,12o)**, which reproduces
`E_FCI = −108.808041914843 Ha` and has 792 α-strings. No proprietary data and no quantum
hardware access is needed for any figure or table: all "real-noise" results use local,
backend-calibrated noise models (`FakeTorino`, IBM Heron r1).

## `gauge_study/` — why the counts are gauge-dependent

The α-string weights are **not invariant** under rotations inside a degenerate orbital
shell: the rotation leaves `E_FCI` untouched and redistributes weight between strings.
This is why the 90 % determinant count is 11 in one run and 12 in another — it is not
anyone's arithmetic error, it is a quantity that is only well defined once the
degenerate multiplets are completed. Six scripts establish this; see
[`gauge_study/README.md`](gauge_study/README.md). The invariant object is the α|β
Schmidt spectrum, verified stable to 10⁻¹⁵ across eight SO(12) rotations while the
string count moves from 10 to 450.

## What is *not* here

Being explicit, so nothing here promises more than it delivers:

- **Figures 5 and 8 have no generator** — they are schematics, drawn in TikZ.
- **Figure 2's orbital panel needs PyMOL**, which is not pip-installable in the
  container above; `render_orb.py` documents the system-Python invocation used.
- **`wf_occ.dat` and `wf_hist.dat` were regenerated on 2026-09-11.** The versions
  shipped with v1 did not come from the declared recipe — they showed the degenerate π
  pairs split (0.6568 / 0.6509) where exact FCI gives them identical (0.6603 / 0.6603),
  and no script reproduced them. They were replaced with data that *is* reproducible
  from the declared recipe, by `gen_fig1.py`. Figure 1 is the workflow schematic; its
  insets illustrate weight concentration and multireference character, both unchanged.

## Layout

```
paper/          main.tex (the live manuscript), main.pdf, its .dat files, arXiv package
figures/        generators of figure data
calculations/   the physics behind the reported numbers
gauge_study/    the six scripts establishing gauge dependence
results/        the verified data files
notebook/       end-to-end Colab notebook
docker/         build environments
versions/       v1 exactly as posted, frozen
```

## Citing

Cite the paper, not the repository — see [`CITATION.cff`](CITATION.cff).

> Bonilla Vargas, N. *Machine learning for sample-based quantum diagonalization:
> generative configuration recovery and the classical-simulability frontier.*
> arXiv:2608.05314 (2026).

Code is MIT; the manuscript text and figures are CC BY 4.0.
