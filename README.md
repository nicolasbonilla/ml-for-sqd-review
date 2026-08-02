# Machine learning for sample-based quantum diagonalization — reproducible code

Reproducible calculations, figures, and the companion notebook for the critical review

> **Machine learning for sample-based quantum diagonalization: generative configuration recovery and the classical-simulability frontier**
> Nicolás Bonilla Vargas — *arXiv:XXXX.XXXXX* (link added on posting)

Every quantitative claim, every figure, and every table in the paper is regenerated **from first principles, with fixed random seeds**, by the code in this repository. Nothing requires proprietary data or quantum-hardware access: all "real-noise" results use *local, backend-calibrated* noise models (`FakeTorino`, IBM Heron r1), so a laptop reproduces everything.

---

### ▶ Run or read the calculations in one click

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolasbonilla/ml-for-sqd-review/blob/main/notebook/GFlowNet_SQD_calculations.ipynb)

- **Read it here, no install:** GitHub renders [`notebook/GFlowNet_SQD_calculations.ipynb`](notebook/GFlowNet_SQD_calculations.ipynb) inline — every cell and every embedded figure is visible in the browser.
- **Run it in the browser:** click the Colab badge above (or [this link](https://colab.research.google.com/github/nicolasbonilla/ml-for-sqd-review/blob/main/notebook/GFlowNet_SQD_calculations.ipynb)) and *Runtime → Run all*.
- **Download the whole repository:** `git clone https://github.com/nicolasbonilla/ml-for-sqd-review.git` — or **Code ▸ Download ZIP** on GitHub. Everything (notebook, calculation scripts, figure scripts, the generated figures in `figures_output/`, and the Docker environments) comes in a single download.
- **See the figures without running anything:** the definitive figures are the native-TikZ ones in the compiled manuscript [`paper/Paper_Review_ML_SQD.pdf`](paper/Paper_Review_ML_SQD.pdf) (source: [`paper/arxiv_source/main.tex`](paper/arxiv_source/main.tex)); [`figures_output/`](figures_output/) additionally holds earlier standalone renders as PDF/PNG.

---

## Quick start

**Option A — one notebook, in the browser (no install).** Open `notebook/GFlowNet_SQD_calculations.ipynb` in Google Colab and run all cells. It reproduces, in order: the exact FCI reference, the coupon-collector statistics, S-CORE recovery under noise, the Epstein–Nesbet reward and the GFlowNet compactness study, the noise crossover, and the decisive heat-bath-CI test.

**Option B — Docker (bit-for-bit).** The exact images used for the paper:
```bash
docker build -t sqd-nb  -f docker/Dockerfile.nb  .   # pyscf + torch + qiskit (calculations, notebook, data-figures)
docker build -t sqd-tex -f docker/Dockerfile.tex .   # texlive-xetex + pdflatex (the manuscript)
docker build -t sqd-viz -f docker/Dockerfile.viz .   # sqd-nb + PyMOL (ray-traced molecular orbitals)
# example: reproduce the decisive classical test (H2O, N2, C2 at D=120)
docker run --rm -v "$PWD:/w" -w /w sqd-nb python calculations/hci_baseline.py
```

**Option C — bare Python (≥3.10).** `pip install -r requirements.txt`, then run any script in `calculations/` or `figures/`.

> **New in the latest revision:** the natural-orbital / order-parameter sweep (`calculations/compute_physics.py`) and the single-reference **H₂O crossover control** (`calculations/h2o_crossover.py`) are standalone analyses that run under the `sqd-nb` image and write to `results/`; the corrected genuine 3σ_g orbital of Fig. 2 is rendered by `figures/gen_sg_true.py` + `figures/render_sg.py` (`sqd-viz`). These are not yet folded into the Colab notebook.

---

## Where every number comes from

| Claim in the paper | Reproduced by | Value |
|---|---|---|
| Exact FCI reference, N₂ CAS(10e,12o) cc-pVDZ, R=2.0 Å | `calculations/mve_backbone.py`, `calculations/verify_coupon.py` | **E = −108.808042 Ha** |
| Coupon-collector: 90 % of weight in 12 of 792 strings (1.5 %) | `calculations/verify_coupon.py` | cum@11 = 0.894, cum@12 = 0.905 |
| Decisive classical test, heat-bath CI at D=120 | `calculations/hci_baseline.py` | H₂O 0.56, N₂ 9.8, C₂ −25.4 mHa (C₂ pathological → dropped) |
| GFlowNet compactness vs classical greedy & oracle (β=0.5 tempered, 5-seed) | `calculations/gflownet_temper.py`, notebook cell 20 | GFlowNet 192±19 vs greedy **41.3** vs oracle 17.0 mHa at D=120 — the cheap-reward GFlowNet does **not** beat the classical greedy selector |
| Order parameter: cheap Epstein–Nesbet reward vs exact \|c\|² (Spearman) | `calculations/compute_physics.py` | ρ ≈ **0.64** at R=2.0 Å; falls monotonically 0.72→0.60 as N₂ stretches into the multireference regime |
| Multireference character: FCI natural-orbital occupations vs geometry | `calculations/compute_physics.py` | HF weight 0.93→0.12; frontier NOONs 1.95/0.06→1.08/0.92 (R=1.1→2.5 Å) |
| Noise crossover, single N₂ geometry | `calculations/noise_sweep.py`, `calculations/gflownet_realnoise.py` | 5-seed gaps [−11.7, +0.3, +9.7, +9.8] mHa (sign-flip floor p=0.0625 at n=5; t is a reproducibility diagnostic) |
| **Controlled test** — crossover vs N₂ geometry (only multireference varies) | `calculations/n2_ladder_crossover.py` | at 3× noise the crossover is **universal** (gap +10…+21 mHa at every R, incl. near-single-reference R=1.1) → driven by shot-starvation, **not** multireference; **refutes** the multireference-specific reading |
| Cross-molecule contrast (confounded) — H₂O vs N₂ | `calculations/h2o_crossover.py` | H₂O 5-seed gaps [−0.1, −1.3, −2.9, +1.4] mHa; the apparent difference is a subspace-coverage confound (24% vs 15% at D=120), not chemistry |
| Backend-calibrated noise (asymmetric readout, FakeTorino / Heron r1) | `calculations/validate_realnoise.py` | — |

## Where every figure comes from

The ten paper figures are **native TikZ / PGFPlots** drawn inside the manuscript source
[`paper/arxiv_source/main.tex`](paper/arxiv_source/main.tex) (the authoritative version; they share the
colour-blind-safe Okabe–Ito palette and keep legends/labels outside the data area). The `figures/` scripts
below generate the underlying **data** (`.dat` in `paper/arxiv_source/`, `results/`) and the ray-traced
orbital panels; the two data-driven exceptions are Fig. 2 (raster orbital renders) and the numeric inputs the
TikZ reads.

| Figure | Data / render source |
|---|---|
| Fig. 1 — SQD workflow | native TikZ (`main.tex`) + data from `figures/data_figs.py` |
| Fig. 2 — active-space molecular orbitals (σ/π/π* + MO ladders) | `figures/compose_fig2.py` + `figures/gen_cubes.py` + `figures/gen_sg_true.py` + `figures/render_orb.py` / `figures/render_sg.py` (PyMOL) |
| Fig. 3 — coupon-collector | native TikZ + `figures/coupon_fig.py` data |
| Fig. 4 — S-CORE recovery under noise | native TikZ + `figures/data_figs.py` (`recovery.dat`) |
| Fig. 5 — taxonomy of ML methods (incl. RL-CI vs GFlowNet whitespace) | native TikZ (`main.tex`) |
| Fig. 6 — GFlowNet compactness | native TikZ + `calculations/gflownet_temper.py` |
| Fig. 7 — decisive heat-bath-CI test | native TikZ + `calculations/hci_baseline.py` |
| Fig. 8 — advantage landscape (§5–§7) | native TikZ (`main.tex`) |
| **Fig. 9 — order parameter, measured** (Spearman ↓ vs multireference ↑) | native TikZ + `calculations/compute_physics.py` (`results/orderparam.dat`) |
| **Fig. 10 — controlled noise-crossover test** (crossover vs N₂ geometry; refutes multireference-specificity) | native TikZ (`results/ladder.dat`) + `calculations/n2_ladder_crossover.py` |

---

## Layout

```
notebook/          GFlowNet_SQD_calculations.ipynb   — end-to-end reproducibility (Colab-ready)
calculations/      MVE system, HCI baseline, coupon-collector, GFlowNet, noise sweep,
                   compute_physics.py (NOONs + order parameter), h2o_crossover.py (2nd system)
figures/           data-generation + ray-traced orbital scripts (native TikZ lives in the manuscript)
results/           computed outputs: orderparam.dat, h2o_crossover.json, physics_results.json
paper/             compiled PDF + markdown-to-LaTeX pipeline
paper/arxiv_source/ authoritative LaTeX source (native-TikZ figures) + the .dat the figures read
docker/            the three exact build environments (sqd-nb, sqd-tex, sqd-viz)
```

## Method, in one paragraph

Sample-based quantum diagonalization (SQD / QSCI) prepares an approximate ground state on a quantum
processor, samples electronic configurations, and diagonalizes the Hamiltonian classically in the
sampled determinant subspace. The efficiency of the loop is set entirely by *which* configurations
enter the subspace — a selection problem for machine learning, made acute by a coupon-collector
bottleneck. This code studies that selection problem on an exactly-solvable minimal viable example
(N₂ / H₂O in a CAS small enough for exact FCI), so every energy is an **error against exact truth**,
and asks — with strong classical baselines, backend-calibrated noise, and multi-seed error bars —
whether a quantum or generative proposer beats classical selected configuration interaction.

## License & citation

Code: MIT (see `LICENSE`). If you use it, please cite the paper (arXiv link above). No proprietary
data or hardware credentials are included or required.

**Contact:** Nicolás Bonilla Vargas — ngbonillav@unal.edu.co
