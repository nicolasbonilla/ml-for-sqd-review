# -*- coding: utf-8 -*-
"""Build a submission-grade LaTeX article from the review markdown draft.
Tailored converter: protects math, maps unicode, escapes LaTeX specials,
turns image markdown into figure floats, injects hand-built tables + refs.
Compile with xelatex."""
import re, io, os
ARXIV = os.environ.get("ARXIV") == "1"   # ARXIV=1 -> pdflatex-compatible preamble for arXiv submission

SRC = "/w/Paper_Review_ML_SQD_borrador.md"
OUT = "/w/paper.tex"
md = io.open(SRC, encoding="utf-8").read()

# ---------------------------------------------------------------- extract regions
def between(text, start, end):
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[i:j].strip()

title = md.splitlines()[0]
title = title.replace("# [BORRADOR — Paper REVIEW] ", "").replace("# ", "").strip()

abstract = between(md, "## ABSTRACT (draft)", "\n---")
intro    = between(md, "## 1. INTRODUCTION (draft completo)", "## ESTRUCTURA").rsplit("\n---", 1)[0].strip()
body     = between(md, "# SECCIONES REDACTADAS", "\n# TABLAS")
refblock = md.split("# REFERENCES", 1)[1]
refblock = refblock.split("\n\n", 1)[1]                       # drop the "(formatted; ...)" header line
refblock = refblock.split("*(Full", 1)[0].strip()

# ---------------------------------------------------------------- inline conversion
UNI = {
    "—": "---", "–": "--", "\u00a0": "~", "…": "\\ldots{}",
    "“": "``", "”": "''", "‘": "`", "’": "'",
    "→": "$\\to$", "←": "$\\leftarrow$", "≈": "$\\approx$", "≃": "$\\simeq$",
    "≤": "$\\le$", "≥": "$\\ge$", "×": "$\\times$", "·": "$\\cdot$", "±": "$\\pm$",
    "∝": "$\\propto$", "∈": "$\\in$", "√": "$\\surd$", "∞": "$\\infty$",
    "≫": "$\\gg$", "≪": "$\\ll$", "≠": "$\\neq$", "†": "$\\dagger$",
    "Å": "\\AA{}", "§": "\\S{}", "①": "\\ent{1}", "②": "\\ent{2}", "③": "\\ent{3}",
    "α": "$\\alpha$", "β": "$\\beta$", "γ": "$\\gamma$", "δ": "$\\delta$",
    "ε": "$\\epsilon$", "θ": "$\\theta$", "Θ": "$\\Theta$", "λ": "$\\lambda$",
    "μ": "$\\mu$", "π": "$\\pi$", "Π": "$\\Pi$", "σ": "$\\sigma$", "φ": "$\\phi$",
    "Φ": "$\\Phi$", "χ": "$\\chi$", "ψ": "$\\psi$", "Ψ": "$\\Psi$", "ω": "$\\omega$",
    "Ω": "$\\Omega$", "∆": "$\\Delta$", "∂": "$\\partial$",
    "₀": "$_0$", "₁": "$_1$", "₂": "$_2$", "₃": "$_3$", "₄": "$_4$", "₅": "$_5$",
    "₆": "$_6$", "₇": "$_7$", "₈": "$_8$", "₉": "$_9$", "ₐ": "$_a$",
    "⁰": "$^0$", "¹": "$^1$", "²": "$^2$", "³": "$^3$", "⁴": "$^4$", "⁵": "$^5$",
    "⁶": "$^6$", "⁷": "$^7$", "⁸": "$^8$", "⁹": "$^9$", "½": "$\\tfrac12$",
}
def esc(t):                        # escape LaTeX specials in plain text
    t = t.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("^", "\\textasciicircum{}")]:
        t = t.replace(a, b)
    return t

def convert_inline(text):
    # 1) protect display math $$...$$ -> equation ; cross-refs [[fig:key]] ; then inline math $...$
    math = []
    def keep(s):
        math.append(s); return f"ZQZ{len(math)-1}ZQZ"
    text = re.sub(r"\$\$(.+?)\$\$",
                  lambda m: keep("\\begin{equation}\n" + m.group(1).strip() + "\n\\end{equation}"),
                  text, flags=re.S)
    text = re.sub(r"\[\[(fig:[^\]]+)\]\]",
                  lambda m: keep("Fig.~\\ref{" + m.group(1) + "}"), text)
    # inline @FIG:key@ (mid-sentence) -> cross-reference; standalone lines are handled upstream
    text = re.sub(r"@FIG:(\w+)@",
                  lambda m: keep("Fig.~\\ref{fig:" + m.group(1) + "}"), text)
    text = re.sub(r"\$[^$]+\$", lambda m: keep(m.group(0)), text)
    # 2) protect inline code `...`
    code = []
    def pc(m):
        code.append(m.group(1)); return f"ZCZ{len(code)-1}ZCZ"
    text = re.sub(r"`([^`]+)`", pc, text)
    # 3) drop markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # 4) escape specials, then tilde (approx) -> $\sim$
    text = esc(text)
    text = text.replace("~", "$\\sim$")
    # 5) unicode map
    for a, b in UNI.items():
        text = text.replace(a, b)
    # 6) bold / italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![A-Za-z0-9])\*(.+?)\*(?![A-Za-z0-9])", r"\\emph{\1}", text)
    # 7) restore code
    for i, c in enumerate(code):
        text = text.replace(f"ZCZ{i}ZCZ", "\\texttt{" + esc(c).replace("~", "\\textasciitilde{}") + "}")
    # 8) restore math
    for i, m in enumerate(math):
        text = text.replace(f"ZQZ{i}ZQZ", m)
    return text

# ---------------------------------------------------------------- figure floats (keyed)
# All quantitative figures are REAL output of the companion notebook (fixed seeds).
FIGDEFS = {
 "loop": ("pf_workflow.pdf", 1.0, "*",
   "\\textbf{The sample-based quantum diagonalization workflow, with real data at every "
   "stage} (N$_2$, CAS(10e,12o), exact-FCI-verifiable). A shallow LUCJ circuit on the quantum "
   "processor (orbital rotations $U(\\theta)$ interleaved with diagonal-Coulomb Jastrow gates) "
   "is sampled in the computational basis; here $76\\%$ of shots survive particle-number "
   "post-selection at Heron-calibrated noise. The classical co-processor repairs the broken "
   "samples by occupation-weighted S-CORE, assembles the recovered determinants into a "
   "subspace $\\mathcal{S}$ (each row of panel~5 is one selected configuration's orbital "
   "occupation), and diagonalizes the projected Hamiltonian, whose energy is a variational "
   "upper bound converging to the exact FCI value (panel~6). Machine learning augments either "
   "the sampler (\\ent{1}, generative circuit design) or the generation/selection of "
   "configurations (\\ent{2}). Every panel is real output of the accompanying calculations."),
 "system": ("pf_system.pdf", 1.0, "*",
   "\\textbf{The molecular systems, their active-space orbital-energy structure, and the "
   "frontier orbitals.} \\emph{(a,f)} Ray-traced ball-and-stick geometries of stretched "
   "N$_2$ ($R=2.0\\,$Å, CAS(10e,12o)) and H$_2$O (CAS(8e,12o)). \\emph{(b,g)} Molecular-"
   "orbital energy-level diagrams (restricted Hartree--Fock, eV): occupied active levels "
   "(blue) carry their two electrons as up/down arrows, virtual active levels are red, the "
   "shaded band is the complete active space, and $E_F$ marks the HOMO--LUMO midpoint; the "
   "deep $1s$ core is frozen and off-scale. \\emph{(c--e)} Ray-traced (PyMOL) isosurfaces of "
   "the N$_2$ $1\\pi_u$ bonding, $3\\sigma_g$ HOMO, and $1\\pi_g^{*}$ LUMO orbitals, with "
   "their energies; the near-degenerate $\\pi/\\pi^{*}$ manifold is what makes stretched "
   "N$_2$ strongly multireference. \\emph{(h,i)} The H$_2$O $1b_1$ lone-pair HOMO and "
   "$4a_1^{*}$ LUMO. Isosurfaces at $|\\psi|=0.028\\,a_0^{-3/2}$; orbital phases orange/blue."),
 "coupon": ("pf_coupon.pdf", 0.98, "*",
   "\\textbf{The coupon-collector bottleneck, computed exactly.} "
   "N$_2$ CAS(10e,12o), from the exact FCI wavefunction. \\emph{Left:} the marginal "
   "single-spin-string weight $w_\\alpha(i)=\\sum_\\beta|c_{i\\beta}|^2$ falls over three "
   "orders of magnitude. \\emph{Right:} $90\\%$ of the ground-state weight is carried by "
   "$12$ of $792$ strings ($1.5\\%$), but the correlation-energy tail is spread over "
   "hundreds of rare strings --- the configurations a sampler must pay to discover."),
 "score": ("pf_recovery.pdf", 0.98, "*",
   "\\textbf{Configuration recovery under backend-calibrated noise} (FakeTorino / Heron r1 "
   "rates; asymmetric readout $+$ depolarizing). \\emph{Left:} the fraction of "
   "particle-number-valid shots collapses as the noise scale grows. \\emph{Right:} S-CORE "
   "recovery restores the subspace energy at fixed dimension $D=120$, turning otherwise-"
   "discarded broken samples back into valid configurations."),
 "taxonomy": ("pf_taxonomy.pdf", 1.0, "*",
   "\\textbf{A taxonomy of machine-learning methods for determinant and subspace selection in "
   "SQD}, organized by the object each method generates (rows) and the importance signal it "
   "exploits (columns). Generative models (purple) learn a distribution over determinants; "
   "discriminative models (orange) classify or rank them; circuit-design methods (green) learn "
   "the sampler itself. The dashed cell marks the methodological whitespace this review "
   "identifies: a reward-proportional generative-flow network that constructs valid-$N$ "
   "determinants in the loop --- unoccupied as of mid-2026."),
 "compact": ("pf_compact.pdf", 0.68, "",
   "\\textbf{Compactness under a cheap, FCI-free reward} (N$_2$, exact FCI). Energy error vs "
   "subspace dimension for blind uniform sampling, deterministic top-$K$ selection by the "
   "static cheap Epstein--Nesbet reward, the GFlowNet trained on the same (tempered) cheap "
   "reward, and the unreachable exact oracle. The GFlowNet learns the important region and "
   "tracks the oracle far better than blind sampling; naive i.i.d.\\ sampling from the peaked "
   "reward saturates below $D=30$ and is off-scale. The GFlowNet curve terminates at "
   "$D{=}120$ because its tempered ($\\beta{=}0.5$) sampler concentrates mass on the "
   "${\\sim}140$ highest-reward determinants --- matching the exact oracle there "
   "(${\\approx}17.7$ vs $17.0$~mHa) while blind uniform sampling stalls near $240$~mHa --- so "
   "covering a larger subspace would call for the temperature schedule that is \\S4.6's "
   "learnable analogue of the CIPSI threshold. The decisive comparison against the "
   "stronger \\emph{iterative} heat-bath CI is Fig.~\\ref{fig:hci}, where the classical "
   "selector wins at every FCI-verifiable scale."),
 "hci": ("pf_hci.pdf", 0.64, "",
   "\\textbf{The decisive classical test at FCI-verifiable scale} ($D=120$). Iterative "
   "heat-bath CI, bootstrapped from the correlated wavefunction, reaches $0.6$~mHa on "
   "H$_2$O and ${\\approx}10$~mHa on stretched N$_2$ --- at or below the noisy quantum-"
   "sampled subspace on every system. At verifiable scale, classical selected CI wins."),
 "crossover": ("pf_crossover.pdf", 0.72, "",
   "\\textbf{The noise crossover.} Energy-error gap between the fair classical control "
   "(S-CORE recovery $+$ cheap-prior fill) and the fused generative proposer, vs device-"
   "noise scale, over random seeds (mean~$\\pm$~s.d.). At low noise the classical control is "
   "better (negative gap); as noise grows the constraint-respecting generative proposer "
   "pulls ahead. The advantage is regime-dependent, not universal."),
 "landscape": ("fig2_nc.pdf", 0.98, "*",
   "\\textbf{The advantage landscape of \\S5--\\S7.} The structure that makes SQD classically "
   "\\emph{verifiable} is the structure that makes it classically \\emph{constructible}; "
   "advantage survives only where that equivalence breaks. Colour-blind-safe (Okabe--Ito) "
   "palette; zones are also distinguished by header text and verdict chips, not colour alone."),
}
def figure_float(key):
    fn, wid, star, cap = FIGDEFS[key]
    env = "figure*" if star else "figure"
    return (f"\n\\begin{{{env}}}[tb]\n\\centering\n"
            f"\\includegraphics[width={wid}\\linewidth]{{{fn}}}\n"
            f"\\caption{{{cap}}}\n\\label{{fig:{key}}}\n\\end{{{env}}}\n")

# ---------------------------------------------------------------- body -> latex
def process_body(text):
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1; continue
        mfig = re.fullmatch(r"@FIG:(\w+)@", ln.strip())  # placement ONLY if the whole line is the marker
        if mfig:
            out.append(figure_float(mfig.group(1)))
            i += 1; continue
        if ln.startswith("!["):                          # legacy image line + its caption -> skip
            if i+1 < len(lines) and lines[i+1].lstrip().startswith("*Figure"):
                i += 1
            i += 1; continue
        if ln.startswith("### "):
            out.append("\n\\subsection{" + convert_inline(re.sub(r"^###\s+[\d.]*\s*", "", ln)) + "}\n")
            i += 1; continue
        if ln.startswith("## "):
            out.append("\n\\section{" + convert_inline(re.sub(r"^##\s+[\d.]*\s*", "", ln)) + "}\n")
            i += 1; continue
        if ln.strip() == "---":
            i += 1; continue
        if ln.lstrip().startswith("*[Fin del borrador"):
            i += 1; continue
        out.append(convert_inline(ln) + "\n")
        i += 1
    return "\n".join(out)

# ---------------------------------------------------------------- references
def process_refs(text):
    entries = [e.strip().replace("\n", " ") for e in re.split(r"\n\s*\n", text.strip()) if e.strip()]
    entries.sort(key=lambda s: re.sub(r"[^A-Za-z]", "", s)[:24].lower())   # alphabetical by author
    out = ["\\begingroup\\footnotesize\\frenchspacing"
           "\\setlength{\\parindent}{-1.1em}\\setlength{\\leftskip}{1.1em}\\setlength{\\parskip}{0.4em}"]
    for e in entries:
        out.append(convert_inline(e) + "\\par")
    out.append("\\endgroup")
    return "\n".join(out)

# ---------------------------------------------------------------- tables (hand-built)
TABLE1 = r"""
\begin{table*}[t]\centering\footnotesize
\caption{The SQD/QSCI method family (curated; the full annotated list is maintained in the state-of-the-art dataset). ``Largest scale'' states hardware qubit counts or active spaces as reported.}
\label{tab:sqd}
\renewcommand{\arraystretch}{1.25}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{2.4cm} >{\raggedright\arraybackslash}p{1.8cm}@{}}
\toprule
\textbf{Category} & \textbf{Method} & \textbf{Key idea} & \textbf{Ref.} & \textbf{Largest scale} \\
\midrule
Core & QSCI & Quantum samples the subspace; classical diagonalization & [1] & 8 qubits \\
 & SQD + S-CORE & Self-consistent configuration recovery; quantum-centric supercomputing & [2] & 77\,q / (54e,36o) \\
Sampler/input & ADAPT-QSCI & Iterative input-state growth, no VQE & [19] & small molecules \\
 & TE-/HSB-QSCI & Time-evolved sampling state & [20] & 36\,q (carbyne) \\
 & Generative-QSCI & Transformer (GQE) designs the sampling circuit & [33] & 32\,q (N$_2$) \\
Krylov & SKQD / SqDRIFT & Sampled Krylov subspace; provable convergence & [21,22] & 49\,q \\
Recovery/select & SQD-AA & Amplitude-amplify unseen bitstrings (quadratic gain) & --- & model systems \\
 & ML recovery & Learned generation/classification (see Table~\ref{tab:ml}) & --- & see Table~\ref{tab:ml} \\
Excited/orbital & ext-/oo-SQD & Extra projection for excited states; orbital optimization & [23] & CH$_2$ 52\,q \\
Symmetry/compact & Compact-QSCI & Space-group symmetry; time-evolved compact subspaces & [41,49] & SiH$_4$ 42\,q \\
Qubit-reduced & HCI-HSQD & Half-qubit; seniority-zero; neural compression & [10] & Fe-S (54e,36o) \\
Embedding & DMET-/LAS-SQD & SQD as fragment / impurity solver & [60] & protein 12{,}635 atoms \\
Hybrid post-proc. & AFQMC-SQD; PT2 & Recover dynamical correlation on the SQD reference & --- & N$_2$; [2Fe-2S] \\
\bottomrule
\end{tabularx}
\end{table*}
"""

TABLE2 = r"""
\begin{table*}[t]\centering\footnotesize
\caption{Machine-learning and generative methods for determinant/subspace selection, organized by the object generated and the importance signal exploited. HW? indicates whether a quantum-hardware demonstration was reported.}
\label{tab:ml}
\renewcommand{\arraystretch}{1.25}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}p{2.5cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{1.55cm} >{\raggedright\arraybackslash}p{2.5cm}@{}}
\toprule
\textbf{Method} & \textbf{Generated object} & \textbf{Importance signal} & \textbf{HW?} & \textbf{Ref.} \\
\midrule
MLCI & determinant (add/skip) & on-the-fly NN prediction & classical & [46] \\
PIGen-SQD & determinants (recovery) & quantum-sample freq.\ + physics prior & IBM Heron & [6] \\
QSCI-RBM (DMET) & determinant distribution & learned Born distribution (RBM) & classical & [24] \\
HI-NQS & determinants (in loop) & PT score + distilled eigenvector & classical GPU & [26] \\
NQS-SC & selected-config energy & variational, on selected set & classical & [27] \\
NNQS-SCI / QiankunNet & determinants & autoregressive Born + de-dup & classical HPC & [28,29] \\
cSQD & determinant label & binary classifier + active learning & classical (+SQD) & [30] \\
HAAR-SCI & determinants & Hamiltonian coupling (gated transformer) & classical GPU & [31] \\
RCI (learning-to-rank) & determinant ranking & pairwise rank (transformer) & classical & [32] \\
Generative-QSCI (GQE) & \textbf{circuit} (not determinants) & QSCI subspace energy (RL policy) & classical sim.\ & [33] \\
\textbf{GFlowNet (proposed)} & \textbf{determinant, valid-$N$ by construction} & \textbf{reward $\propto$ fused EN + sample freq.} & --- (whitespace) & \emph{this work, \S4.6} \\
\bottomrule
\end{tabularx}
\end{table*}
"""

# ---------------------------------------------------------------- assemble
PREAMBLE = r"""\documentclass[11pt]{article}
FONTPKG
\usepackage{amsmath,amssymb}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{titlesec}
\usepackage{caption}
\usepackage[dvipsnames]{xcolor}
\usepackage[colorlinks=true,allcolors=Blue,breaklinks=true]{hyperref}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\titleformat{\section}{\normalfont\large\bfseries}{\thesection.}{0.6em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
\newcommand{\ent}[1]{\raisebox{0.1ex}{\textcircled{\scriptsize #1}}}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
\frenchspacing
\sloppy
\hyphenpenalty=1200
\title{\vspace{-1.2cm}\bfseries TITLEHERE}
\author{Nicol\'as Bonilla Vargas\thanks{M.Sc.\ Physics, Universidad Nacional de Colombia; M.Sc.\ Applied Data Science \& AI, SRH University Heidelberg / Munich. \texttt{ngbonillav@unal.edu.co}}}
\date{\today}
"""
PREAMBLE = PREAMBLE.replace("FONTPKG",
    "\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}" if ARXIV
    else "\\usepackage{fontspec}")

doc = []
doc.append(PREAMBLE.replace("TITLEHERE", convert_inline(title)))
doc.append("\\begin{document}\n\\maketitle")
doc.append("\\begin{abstract}\n" + convert_inline(abstract) + "\n\\end{abstract}")
doc.append("\n\\section{Introduction}\n")
for para in intro.split("\n"):
    if para.strip():
        doc.append(convert_inline(para) + "\n")
body = body.replace("[ref: Estado_del_Arte]", "(see the accompanying dataset)")
# convert numeric [n] citations in the hand-built tables to author-year (unify with the body)
_NUM2AY = {"1":"Kanno et al. 2023","2":"Robledo-Moreno et al. 2025","6":"Patra et al. 2025",
 "10":"McFarthing et al. 2026","19":"Nakagawa et al. 2024","20":"Mikkelsen--Nakagawa 2025",
 "21":"Yu et al. 2025","22":"Piccinelli et al. 2025","23":"Barison et al. 2025","24":"Patra et al. 2026",
 "26":"Chang et al. 2026","27":"Solanki--Ding--Reiher 2026","28":"Shang et al. 2025","29":"Sun et al. 2026",
 "30":"Zeni et al. 2026","31":"Zhang et al. 2025","32":"Nie et al. 2026","33":"Kemmoku et al. 2026",
 "41":"Weaving et al. 2025","46":"Coe 2018","49":"Nogaki et al. 2026","60":"Wang et al. 2026"}
def _num2ay(s):
    return re.sub(r"\[([0-9]+(?:,[0-9]+)*)\]",
                  lambda m: "[" + "; ".join(_NUM2AY.get(x, x) for x in m.group(1).split(",")) + "]", s)
body_tex = process_body(body)
# place the tables as floats near where they are first referenced (Table 1 in §2, Table 2 in §4),
# instead of dumping them at the end of the document
body_tex = body_tex.replace(
    "\\subsection{Hardware demonstrations and honest scale}",
    _num2ay(TABLE1) + "\n\\subsection{Hardware demonstrations and honest scale}", 1)
body_tex = body_tex.replace(
    "\\subsection{Restricted Boltzmann machines for generative recovery}",
    _num2ay(TABLE2) + "\n\\subsection{Restricted Boltzmann machines for generative recovery}", 1)
doc.append(body_tex)

BACKMATTER = r"""
\section*{Data and code availability}
\small
Every quantitative claim in this review is reproduced from executable code in the companion computational
notebook \texttt{GFlowNet\_SQD\_calculations.ipynb}, which runs in Google Colab or any Python~$\ge$3.10 with
\texttt{pyscf}, \texttt{torch}, \texttt{scipy}, and \texttt{matplotlib}. The notebook regenerates, from first
principles and with fixed random seeds, the exact FCI reference of Fig.~1's system, the coupon-collector
statistics of \S3, the S-CORE recovery of \S3.2, the Epstein--Nesbet reward and GFlowNet compactness study of
\S4, the noise-crossover of \S6, and the heat-bath-CI decisive test of \S5. Figures~1 and~2 were produced as
vector graphics and are available with the source. No proprietary data or hardware access is required to
reproduce any figure or table.

\section*{Author contributions}
\small
N.B.V.\ conceived the review, performed the literature synthesis, designed and ran all computational
experiments, produced the figures, and wrote the manuscript.

\section*{Competing interests}
\small
The author declares no competing financial or non-financial interests.

\section*{Acknowledgements}
\small
The author thanks the IBM Quantum and Qiskit communities and the organizers of the Qiskit Global Summer School,
whose open materials seeded this line of work, and acknowledges the developers of \texttt{PySCF} and the
open-source scientific-Python ecosystem on which the accompanying calculations depend.
"""
doc.append(BACKMATTER)
doc.append("\n\\section*{References}\n\\small\n" + process_refs(refblock))
doc.append("\n\\end{document}\n")

io.open(OUT, "w", encoding="utf-8").write("\n".join(doc))
print("wrote", OUT, "chars:", sum(len(x) for x in doc))
