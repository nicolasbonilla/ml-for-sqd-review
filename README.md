<div align="center">

# Machine learning for sample-based quantum diagonalization
### *a review of generative configuration recovery and the classical-simulability frontier*

**Nicolás Bonilla Vargas** &nbsp;[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--6155--4391-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0006-6155-4391)

[![arXiv](https://img.shields.io/badge/arXiv-2608.05314-b31b1b.svg)](https://arxiv.org/abs/2608.05314)
[![Paper](https://img.shields.io/badge/paper-PDF%20(41%20pp)-blue.svg)](paper/main.pdf)
[![Type](https://img.shields.io/badge/type-review%20%2F%20perspective-8A2BE2.svg)](paper/main.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22827270.svg)](https://doi.org/10.5281/zenodo.22827270)
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
| 2 `fig:system` — molecular orbitals | `pf_system.pdf` | [`figures/compose_fig2.py`](figures/compose_fig2.py), which needs **four** upstream steps: `gen_cubes.py` and `gen_sg_true.py` write the cubes, then `render_orb.py` and `render_sg.py` turn them into PNGs (both PyMOL, in `docker/Dockerfile.viz`). Panel (d) loads `o_n2_sg_true.png` specifically — the genuine 3σ_g, MO 4 — so skipping `gen_sg_true.py`/`render_sg.py` aborts there. |
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

## Checking this deposit against the paper, mechanically

The manuscript's §6 argues that a deposit must be able to regenerate what it
accompanies. The failure mode that argument is really about is quieter than a missing
script: a figure gets regenerated and the sentence describing it does not, or the
reverse, and nothing complains. That is how this repository drifted, repeatedly.

[`calculations/verifica_deposito.py`](calculations/verifica_deposito.py) closes the loop.
It reads `paper/main.tex`, extracts the values actually typeset, reads the data files,
and compares them. It holds **no second copy of the numbers** — both sides are read at
run time, so a check can only pass if the manuscript and the deposit agree. It needs no
PySCF and takes under a second:

```bash
python calculations/verifica_deposito.py     # exit 0 = they agree, 1 = they do not
```

A check that fails with *"el patron no aparece en main.tex"* is not a false alarm: it
means a sentence was reworded and this file was not updated with it, which is the first
step of every drift this repository has had.

## Two traps that will cost you a day

**1 · The reward floor of Fig. 6 is a real knob — and it does *not* decide the winner.**
`np.maximum(w, 1e-12)` versus `FLOOR = 1e-3 * w.max()` is nine orders of magnitude, and it
moves the GFlowNet's error at `D = 120` by more than a factor of two. v2 **ran** the floor
instead of choosing it — three settings × five seeds, everything else held fixed
([`figures/barrido_suelo.py`](figures/barrido_suelo.py); raw output in
[`results/`](results/)):

| reward floor | GFlowNet @ `D=120` | greedy top-*K* @ `D=120` | who wins |
|---|---|---|---|
| `1e-3 · max` | 110.8 ± 11.0 mHa | 46.4 mHa | greedy |
| `1e-6 · max` | 58.2 ± 6.6 mHa | 46.4 mHa | greedy |
| `1e-12 · max` | **cannot fill `D=120`** | 46.4 mHa | greedy, by default |

Lowering the floor improves the proposer monotonically but never overtakes the static
ranking at this dimension, and below some point it can no longer *fill* the subspace at
all: only **91 of the 792 strings carry non-zero cheap reward**
([`figures/cuenta_rankeables.py`](figures/cuenta_rankeables.py)), so an unfloored sampler
cannot emit more distinct strings than the reward can rank. Across the entire sweep the
proposer overtakes greedy in **exactly one cell** — `D = 60` at `1e-12`, 51.6 against
54.6 mHa, unresolved at five seeds. The floor is the exploration–exploitation trade-off
made explicit, not a tuning knob with a lucky setting.

> ⚠️ **This README used to say the floor "reverses the conclusion."** That was written
> from a single setting, before the sweep existed. It is wrong, and the sweep above is
> what replaced it. Paper § 4.6 states the corrected version.

The v1 script `compact_fig.py` still does *not* reproduce Fig. 6 — it floors at `1e-12` —
and is kept only as
[`figures/OBSOLETO_compact_fig_v1.py`](figures/OBSOLETO_compact_fig_v1.py) with a warning
header. **Use `fig6_5seed.py`.**

**2 · `%` is a comment, `#` is not — and the column-name row is neither.** pgfplots
does **not** treat `#` as a comment character: a `#`-commented header is read as data and
silently destroys the plot with no compilation error. Every `.dat` here therefore comments
its provenance header with `%`.

But the row of **column names** (`R rho whf …`) is deliberately *not* commented, because
pgfplots needs it for `\addplot table[x=R, y=gap1]` — and `%` *is* a comment to pgfplots,
so commenting it would hide the names. That means the obvious numpy call fails:

```python
np.loadtxt("results/ladder.dat", comments="%")
# ValueError: could not convert string 'R' to float64
```

`np.genfromtxt(..., comments='%', names=True)` fails too, and `skiprows` does not
rescue `loadtxt` the way you would expect: numpy counts comment lines *inside* `skiprows`,
so `skiprows=1` skips the first `%` line, not the column names.

Two recipes that do work. Both are tested against every `.dat` in this repository:

```python
# 1 — keep the column names (this is what the companion notebook does)
import numpy as np
rows = [l.split() for l in open("results/ladder.dat", encoding="utf-8")
        if l.strip() and not l.lstrip().startswith("%")]
cols = {h: np.array([float(r[k]) for r in rows[1:]]) for k, h in enumerate(rows[0])}
cols["gap1"]          # -> array([-4.028, 0.34, 3.011, 5.231])

# 2 — plain array, if you only want the numbers
n_hdr = sum(1 for l in open("results/ladder.dat", encoding="utf-8")
            if l.lstrip().startswith("%"))
data = np.loadtxt("results/ladder.dat", comments="%", skiprows=n_hdr + 1)
```

Earlier versions of this README told you to use `comments='%'` alone. That does not work
on any file here, and it had never been run.

## Run it in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolasbonilla/ml-for-sqd-review/blob/main/notebook/GFlowNet_SQD_calculations.ipynb)

The companion notebook runs end-to-end on a free Colab CPU runtime; uncomment the
`pip install` in its first cell. It loads the production data files straight from this
repository, so the numbers it prints beside its own live run are the deposited ones.

## Reproducing

PySCF publishes no Windows wheels, so on Windows it is Docker or WSL. **Two images,
and which one you need depends on the script.**

The light one is enough for anything that only touches PySCF — `gen_fig1.py`,
`compute_physics.py`, `hci_baseline.py`, and every script in `gauge_study/`:

```bash
printf 'FROM python:3.11-slim\nRUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*\nRUN pip install --no-cache-dir numpy scipy pyscf\n' > Dockerfile
docker build -t sqd-fci .
docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-fci python /w/figures/gen_fig1.py
```

**It is not enough for the GFlowNet scripts.** `figures/barrido_suelo.py`,
`figures/fig6_5seed.py`, `calculations/n2_ladder_crossover.py` and
`figures/data_figs.py` import `torch` or `matplotlib`, neither of which the recipe
above installs; run as written they stop at `ModuleNotFoundError`. Use the deposited
image for those, which also carries `nbconvert` for the notebook:

```bash
docker build -t sqd-nb -f docker/Dockerfile.nb .
docker run --rm -e OMP_NUM_THREADS=1 -v "$PWD":/w -w /w sqd-nb \
    python /w/figures/barrido_suelo.py           # FLOOR_REL=1e-3 by default
docker run --rm -e OMP_NUM_THREADS=1 -e FLOOR_REL=1e-12 -v "$PWD":/w -w /w sqd-nb \
    python /w/figures/barrido_suelo.py
```

Earlier versions of this README printed only the light recipe. It cannot run four of
the generators listed above, and that was not tested.

> **Where the generators write.** The scripts under `calculations/` and `figures/` were
> written to run inside the container with the repository mounted at `/w`, and most of
> the older ones write their output to that **mount root**, not to `results/`. Running
> them as documented therefore leaves `ladder.dat`, `wf_occ.dat`, `orderparam.dat` and
> friends sitting in the repository root, and you have to move them into `results/` and
> `paper/` yourself. The newer ones — `lectura_simetrica.py`, `gfn_ruidoso_fig7.py`,
> `hci_baseline.py`, `barrido_suelo.py` — write straight into `results/`. This asymmetry
> is how the deposit accumulated files their declared generator did not reproduce, so it
> is worth knowing before you conclude that a regeneration "did nothing".

**Pin the threads.** With degenerate shells, canonical RHF orbitals are *not*
reproducible run to run. One set of four runs with default threading gave π orientations
of 30.05° / 13.83° / 64.94° / 144.92°; a later set of five gave 85.5° / 29.2° / 169.8° /
92.3° / 140.2°. **Both are correct, and neither is repeatable** — that is the finding, not
a discrepancy between them. With `OMP_NUM_THREADS=1` every run gives 130.192°, coefficients
identical to nine decimals; with `symmetry=True`, exactly 90.000°. All of the above are
reproduced by [`gauge_study/determinismo.py`](gauge_study/determinismo.py). Run with no
argument it does the canonical draw; the symmetry-adapted 90.000° needs `determinismo.py
sym`. (It used to require the argument in both cases and crash without it.)

The system throughout is **N₂, R = 2.0 Å, cc-pVDZ, CAS(10e,12o)**, which has 792
α-strings and reproduces `E_FCI = −108.808041914843 Ha`.

> **That last digit is not the same in every deposited file, and the reason is convergence,
> not physics.** Files produced with the tight tolerances their headers declare
> (`scf.conv_tol=1e-12`, `fci.conv_tol=1e-13` — `coupon_w.dat`, `coupon_cum.dat`,
> `wf_occ.dat`) give `−108.8080419148…43` or `…42`; files produced at PySCF defaults
> (`physics_results.json`, `recovery.dat`, `fig6_suelo_*.json`) give `−108.80804191435…`.
> The spread is 4.9 × 10⁻¹⁰ Ha. It is physically nil, but it is **not** negligible against
> the 4 × 10⁻¹³ Ha the paper quotes for gauge invariance, so the two should not be compared
> without noticing which file each came from. Tighten the tolerances if you need the last
> three digits — and note that the 90 % string count needs them: at defaults the degenerate
> π pair agrees only to ~10⁻⁷, the multiplet is not recognised, and the count comes out 11
> instead of 12. No proprietary data and no quantum
hardware access is needed for any figure or table: all "real-noise" results use local,
backend-calibrated noise models (`FakeTorino`, IBM Heron r1).

## `gauge_study/` — why the counts are gauge-dependent

The α-string weights are **not invariant** under rotations inside a degenerate orbital
shell: the rotation leaves `E_FCI` untouched and redistributes weight between strings.
This is why the 90 % determinant count is 11 in one run and 12 in another — it is not
anyone's arithmetic error, it is a quantity that is only well defined once the
degenerate multiplets are completed. Nine scripts establish this; see
[`gauge_study/README.md`](gauge_study/README.md). The invariant object is the α|β
Schmidt spectrum, verified stable to 10⁻¹⁵ across eight SO(12) rotations while the
string count moves from 10 to 450.

## Beyond the figures: what else the paper's claims rest on

Fourteen claims in the manuscript are not figure data and would otherwise have no generator.
They do now.

| claim | § | produced by |
|---|---|---|
| Reward-floor sweep — greedy wins at every floor that can fill `D=120` | 4.6 | [`figures/barrido_suelo.py`](figures/barrido_suelo.py) → `results/fig6_suelo_*.json` |
| Only 91 of 792 strings carry non-zero cheap reward | 4.6 | [`figures/cuenta_rankeables.py`](figures/cuenta_rankeables.py) |
| C₂ at 1.24 Å: multiplet completion *widens* the 90 % range, 4–5 → 4–6 | 3.1 | [`gauge_study/c2_multiplete.py`](gauge_study/c2_multiplete.py) |
| The 1× noise ladder does not replicate at an independent five seeds | 7 | [`calculations/n2_ladder_crossover.py`](calculations/n2_ladder_crossover.py) with `SEEDS=5,6,7,8,9` → `results/ladder_replicacion_semillas5-9.dat` |
| The controlled ladder — `.dat` and `.json` from one run, verified reproducible | 7 | [`calculations/n2_ladder_crossover.py`](calculations/n2_ladder_crossover.py) |
| How much the threshold and the orbital gauge each move ρ | 7 | [`gauge_study/rho_sensibilidad.py`](gauge_study/rho_sensibilidad.py) |
| The orbit counts, the invariant fraction, and the count spread under rotation | 3.1 | [`gauge_study/bloque_conteos.py`](gauge_study/bloque_conteos.py) |
| The symmetric-readout control that reverses the ordering | 7 | [`calculations/lectura_simetrica.py`](calculations/lectura_simetrica.py) → `results/lectura_simetrica.json` |
| The dimension at which N₂ enters the 1.3 mHa band, and the selector-vs-ranking sweep over fourteen points | 5.1 | [`calculations/rejilla_D.py`](calculations/rejilla_D.py) → `results/rejilla_D.json` |
| The like-for-like control: sampling a reward costs more than ranking it | 4.6 | [`figures/control_iid.py`](figures/control_iid.py) → `results/control_iid.json` |
| How far the coupon-collector bound overstates the true collection cost (2.5×, 5.8×, 16×) | 3.1 | [`calculations/cota_coleccionista.py`](calculations/cota_coleccionista.py) → `results/cota_coleccionista.json` |
| The 12-qubit and 24-qubit shot-survival fractions of Fig. 1 | 2.1 | [`calculations/supervivencia_fig1.py`](calculations/supervivencia_fig1.py) → `results/supervivencia_fig1.json` |
| **That the abstract says the same thing in all four places it lives, and fits arXiv's 1920-character limit** | — | [`calculations/sincro_resumen.py`](calculations/sincro_resumen.py) |
| **That the checked subset of the numbers above still matches what the paper prints** | 6 | [`calculations/verifica_deposito.py`](calculations/verifica_deposito.py) |

## What is *not* here

Being explicit, so nothing here promises more than it delivers:

- **Figures 5 and 8 have no generator** — they are schematics, drawn in TikZ.
- **Figure 2's orbital panel needs PyMOL**, which is not pip-installable in the
  container above; `render_orb.py` documents the system-Python invocation used.
- **`wf_occ.dat` and `wf_hist.dat` were regenerated on 2026-09-11, and again on
  2026-09-13 in a declared gauge.** The versions shipped with v1 did not come from the
  declared recipe — they showed the degenerate π pairs split (0.6568 / 0.6509) where exact
  FCI gives them identical (0.6603 / 0.6603), and no script reproduced them. The September 11
  replacement *was* reproducible, but only as a side effect of running single-threaded:
  `gen_fig1.py` did not pin `symmetry=True`, and `wf_hist.dat` holds **string counts**,
  which §2.1 of the paper is entirely about being gauge-dependent. It is pinned now.
  (`wf_occ.dat` does not move either way: 1-RDM occupations are invariant inside a
  degenerate shell in any basis of that shell.) Figure 1 is the workflow schematic; its
  insets illustrate weight concentration and multireference character.

- **Figure 2's orbital panel is *not* in the symmetry-adapted gauge, and cannot be
  regenerated in this version.** `gen_cubes.py` runs canonical, so the π orbitals drawn
  are one draw from a family that intra-shell rotation moves; the irreducible representation
  labels are unaffected, and with `OMP_NUM_THREADS=1` the draw is the deterministic 130.192°
  one. We declare it rather than change it, because re-rendering the panel would change a
  figure in a version whose other changes are all corrections. It *is* regenerable:
  `docker/Dockerfile.viz` installs PyMOL, and `render_orb.py` documents the invocation.
  (An earlier draft of this README said PyMOL was not installable here. That was wrong.)

## Layout

```
paper/          main.tex (the live manuscript), main.pdf, its .dat files, arXiv package
                resync_arxiv.py rebuilds that package from main.tex, compiles the
                EXTRACTED zip (not the working directory), and refreshes main.pdf
figures/        generators of figure data
calculations/   the physics behind the reported numbers
gauge_study/    the nine scripts establishing gauge dependence
docs/           the arXiv-form abstract
results/        the verified data files
notebook/       end-to-end Colab notebook
docker/         build environments
versions/       v1 exactly as posted, frozen
```

## Citing

Cite the paper, not the repository — see [`CITATION.cff`](CITATION.cff).

> Bonilla Vargas, N. *Machine learning for sample-based quantum diagonalization:
> a review of generative configuration recovery and the classical-simulability frontier.*
> arXiv:2608.05314 (2026).

The title gained the words *a review of* in v2. Quantum requires textually that
*"reviews should contain the word 'review' in the title"*; arXiv permits a title change on
a replacement. v1 is frozen under [`versions/`](versions/) under its original title.

Code is MIT; the manuscript text and figures are CC BY 4.0.
