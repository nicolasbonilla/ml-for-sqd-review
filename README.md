# Machine learning for sample-based quantum diagonalization — reproducible code

Reproducible calculations, figures, and the companion notebook for the critical review

> **Machine learning for sample-based quantum diagonalization: generative configuration recovery and the classical-simulability frontier**
> Nicolás Bonilla Vargas — *arXiv:XXXX.XXXXX* (link added on posting)

Every quantitative claim, every figure, and every table in the paper is regenerated **from first principles, with fixed random seeds**, by the code in this repository. Nothing requires proprietary data or quantum-hardware access: all "real-noise" results use *local, backend-calibrated* noise models (`FakeTorino`, IBM Heron r1), so a laptop reproduces everything.

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

---

## Where every number comes from

| Claim in the paper | Reproduced by | Value |
|---|---|---|
| Exact FCI reference, N₂ CAS(10e,12o) cc-pVDZ, R=2.0 Å | `calculations/mve_backbone.py`, `calculations/verify_coupon.py` | **E = −108.808042 Ha** |
| Coupon-collector: 90 % of weight in 12 of 792 strings (1.5 %) | `calculations/verify_coupon.py` | cum@11 = 0.894, cum@12 = 0.905 |
| Decisive classical test, heat-bath CI at D=120 | `calculations/hci_baseline.py` | H₂O 0.56, N₂ 9.8, C₂ −25.4 mHa (C₂ pathological → dropped) |
| GFlowNet compactness vs oracle (β=0.5 tempered reward) | `calculations/gflownet_temper.py`, notebook cell 20 | tracks oracle to D=120 |
| Cheap Epstein–Nesbet reward vs true CI weights | `calculations/gflownet_cheap.py` | log-rank-correlation ≈ 0.55 |
| Noise crossover (fair classical control vs fused generative) | `calculations/noise_sweep.py`, `calculations/gflownet_realnoise.py` | 5-seed gaps [−11.7, +0.3, +9.7, +9.8] mHa |
| Backend-calibrated noise (asymmetric readout, FakeTorino / Heron r1) | `calculations/validate_realnoise.py` | — |

## Where every figure comes from

| Figure | Script |
|---|---|
| Fig. 1 — SQD workflow | `figures/workflow_fig.py` |
| Fig. 2 — active-space molecular orbitals | `figures/compose_fig2.py` + `figures/gen_cubes.py` + `figures/render_orb.py` (PyMOL) |
| Fig. 3 — coupon-collector | `figures/coupon_fig.py` |
| Fig. 4 — S-CORE recovery under noise | `figures/data_figs.py` |
| Fig. 5 — taxonomy of ML methods | `figures/taxonomy_fig.py` |
| Fig. 6 — GFlowNet compactness | `figures/compact_fig.py` |
| Fig. 7 — decisive heat-bath-CI test | `figures/hci_fig.py` |
| Fig. 8 — advantage landscape (§5–§7) | `figures/render_nc.py` + `figures/fig2_nc.html` |
| Fig. 9 — noise crossover | `figures/crossover_fig.py` |

All figures use the colour-blind-safe Okabe–Ito palette and place legends/labels outside the data area.

---

## Layout

```
notebook/      GFlowNet_SQD_calculations.ipynb   — end-to-end reproducibility (Colab-ready)
calculations/  MVE system, HCI baseline, coupon-collector, GFlowNet, noise sweep
figures/       one script per paper figure (+ HTML/SVG sources for the vector panels)
docker/        the three exact build environments
paper/         markdown-to-LaTeX build pipeline used to typeset the manuscript
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
