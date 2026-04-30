# Model specification and pipeline reference

This document gives the full mathematical specification of the model implemented in this repository, the file-to-component mapping, and the canonical run order with the corresponding outputs.

GitHub renders the LaTeX in this file natively (MathJax). For a local preview use any markdown viewer with math support (VSCode, Obsidian, `pandoc --mathjax`).

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $t \in \{1,\dots,T\}$ | trial index, $T = 180$ |
| $c \in \{f, v\}$ | cue stream: $f$ = facial, $v$ = verbal |
| $g_t \in \{1,2,3\}$ | facial equivalence group (EG) presented on trial $t$ |
| $u_{c,t}\in\{0,1\}$ | input: cue $c$ predicted the correct option on trial $t$ |
| $d_{c,t}\in\{L,R\}$ | direction in which cue $c$ points on trial $t$ |
| $y_t\in\{0,1\}$ | response: $1$ if participant chose left |
| $\mathbf{x}_{c,t}=(x_{1,c,t},x_{2,c,t},x_{3,c,t})$ | hidden states (Levels 1–3) for stream $c$ |
| $\mu_{k,c,t},\;\sigma^2_{k,c,t}$ | posterior mean and variance at level $k$ |
| $\hat\mu_{k,c,t},\;\hat\pi_{k,c,t}$ | predictive (pre-update) mean and precision |
| $\delta_{k,c,t}$ | prediction error at level $k$ |
| $\kappa_c$ | volatility-to-learning coupling, $\kappa_c\in[0,1]$ |
| $\omega_c$ | tonic Level-2 log-volatility, fixed at $\omega_c=-4$ |
| $\vartheta_c$ | meta-volatility (Level-3 step variance), $\vartheta_c\in[0,1]$ |
| $\zeta$ | facial-vs-verbal cue-weighting bias (reported in log space) |
| $\beta$ | inverse temperature |
| $\gamma$ | (M4 only) cross-EG generalisation strength, $\gamma\in[0,1]$ |
| $s(\cdot)$ | logistic sigmoid, $s(x)=\bigl(1+e^{-x}\bigr)^{-1}$ |
| $\mathbb{1}[\cdot]$ | indicator function |

Free parameters of the winning model (Architecture A):
$\boldsymbol\theta = (\kappa_f,\vartheta_f,\kappa_v,\vartheta_v,\zeta,\beta).$

---

## 2. Perceptual model — three-level binary HGF per cue stream

For each stream $c\in\{f,v\}$ a three-level Hierarchical Gaussian Filter tracks (1) the trial outcome, (2) the cue-accuracy belief, and (3) phasic volatility. The generative process is

$$
\begin{aligned}
u_{c,t} \mid x_{2,c,t} &\sim \mathrm{Bernoulli}\!\bigl(s(x_{2,c,t})\bigr), \\
x_{2,c,t} \mid x_{2,c,t-1},x_{3,c,t-1} &\sim \mathcal{N}\!\bigl(x_{2,c,t-1},\; \exp(\kappa_c\,x_{3,c,t-1}+\omega_c)\bigr), \\
x_{3,c,t} \mid x_{3,c,t-1} &\sim \mathcal{N}\!\bigl(x_{3,c,t-1},\; \vartheta_c\bigr).
\end{aligned}
$$

### 2.1 Variational update (Mathys et al., 2011; 2014)

At each trial, the agent performs a one-step variational update under a Gaussian (Laplace) approximation to the posterior. With $\hat\mu_{k,c,t}=\mu_{k,c,t-1}$ and $\hat\pi_{k,c,t}=1/(\sigma^2_{k,c,t-1}+\nu_{k,c,t})$,

$$
\nu_{2,c,t}=\exp(\kappa_c\,\mu_{3,c,t-1}+\omega_c),\qquad
\nu_{3,c,t}=\vartheta_c.
$$

**Level 1 prediction error** (Bernoulli):
$$
\delta_{1,c,t} \;=\; u_{c,t}-s(\hat\mu_{2,c,t}).
$$

**Level 2 update.** With Bernoulli precision $\hat\pi_{1,c,t}=s(\hat\mu_{2,c,t})\bigl(1-s(\hat\mu_{2,c,t})\bigr)$,
$$
\pi_{2,c,t} \;=\; \hat\pi_{2,c,t}+\hat\pi_{1,c,t}, \qquad
\mu_{2,c,t} \;=\; \hat\mu_{2,c,t}+\frac{1}{\pi_{2,c,t}}\,\delta_{1,c,t}.
$$

**Level 2 volatility prediction error** $\delta_{2,c,t}$ and weights $w_{2,c,t}, r_{2,c,t}$:
$$
\delta_{2,c,t}=\frac{\sigma^2_{2,c,t}+\bigl(\mu_{2,c,t}-\hat\mu_{2,c,t}\bigr)^2}{\sigma^2_{2,c,t-1}+\nu_{2,c,t}}-1,
$$
$$
w_{2,c,t}=\frac{\nu_{2,c,t}}{\sigma^2_{2,c,t-1}+\nu_{2,c,t}}, \qquad
r_{2,c,t}=\frac{\nu_{2,c,t}-\sigma^2_{2,c,t-1}}{\sigma^2_{2,c,t-1}+\nu_{2,c,t}}.
$$

**Level 3 update.**
$$
\pi_{3,c,t}=\hat\pi_{3,c,t}+\tfrac{1}{2}\kappa_c^{2}\,w_{2,c,t}^{2}\bigl(1+r_{2,c,t}\,\delta_{2,c,t}\bigr)-\tfrac{1}{2}\kappa_c^{2}\,w_{2,c,t}^{2}\,\delta_{2,c,t},
$$
$$
\mu_{3,c,t}=\mu_{3,c,t-1}+\frac{\kappa_c}{2\,\pi_{3,c,t}}\,w_{2,c,t}\,\delta_{2,c,t}.
$$

The coupling $\kappa_c$ is the central quantity in HA1: it scales how strongly perceived Level-3 volatility expands the effective Level-2 step size $\nu_{2,c,t}$ and therefore the learning rate.

### 2.2 Architecture variants over the facial cue

The verbal stream is identical across all four architectures. Only the facial side differs.

**Architecture A (Unified, baseline).** A single facial HGF that ignores the EG identifier:
$$
\forall\,t,\quad \text{update facial states using } u_{f,t}\text{ regardless of }g_t.
$$

**Architecture B (Separated).** Three facial HGFs $\{\mathbf{x}^{(g)}_{f,t}\}_{g=1}^{3}$ sharing $(\kappa_f,\omega_f,\vartheta_f)$. Only the active EG is updated:
$$
\mathbf{x}^{(g)}_{f,t}=\begin{cases}\text{HGF update with }u_{f,t},& g=g_t,\\[2pt]\mathbf{x}^{(g)}_{f,t-1},& g\neq g_t.\end{cases}
$$

**Architecture C (Hybrid).** A *shared* Level-3 volatility state across EGs but *separate* Level-2 accuracy beliefs:
$$
x_{3,f,t}\text{ shared},\qquad x_{2,f,t}^{(g)}\text{ EG-specific},\qquad g\in\{1,2,3\}.
$$
The active EG's Level-2 prediction error drives the shared $x_{3,f,t}$ update; inactive EGs' Level-2 states carry forward but inherit the updated shared volatility on their next prediction.

**Model M4 (Generalisation strength $\gamma$).** Architecture A extended with a free $\gamma\in[0,1]$ that scales the Level-2 update applied to the shared facial belief:
$$
\mu_{2,f,t}=\hat\mu_{2,f,t}+\gamma\,\frac{1}{\pi_{2,f,t}}\,\delta_{1,f,t}.
$$
$\gamma\to 1$ recovers Arch A; $\gamma\to 0$ approaches behavioural independence across EGs.

### 2.3 Initial-condition (I1) variants

Three further variants of Architecture A free the Level-2 priors:

| Variant | Free initials | # extra perceptual params |
| --- | --- | --- |
| `I1_Mean` | $\mu_{2,f,0},\;\mu_{2,v,0}$ | $+2$ |
| `I1_Precision` | $\sigma^2_{2,f,0},\;\sigma^2_{2,v,0}$ | $+2$ |
| `I1_Both` | $\mu_{2,f,0},\sigma^2_{2,f,0},\mu_{2,v,0},\sigma^2_{2,v,0}$ | $+4$ |

---

## 3. Response model — precision-weighted integration

Let $\hat b_{c,t}=s(\hat\mu_{2,c,t})$ be the predicted probability that cue $c$ correctly indicates the higher-value option. The probability that cue $c$ points to the *left* being correct is
$$
\hat b^{L}_{c,t} \;=\; \mathbb{1}[d_{c,t}=L]\,\hat b_{c,t} + \mathbb{1}[d_{c,t}=R]\,\bigl(1-\hat b_{c,t}\bigr).
$$

**Cue precisions** (Fisher information of the Bernoulli at $\hat b_{c,t}$):
$$
\pi^{\,\mathrm{cue}}_{c,t} \;=\; \bigl[\hat b_{c,t}\,(1-\hat b_{c,t})\bigr]^{-1}.
$$

**Weights** (with native-space $e^\zeta$ on facial precision):
$$
w_{f,t}=\frac{e^{\zeta}\pi^{\,\mathrm{cue}}_{f,t}}{e^{\zeta}\pi^{\,\mathrm{cue}}_{f,t}+\pi^{\,\mathrm{cue}}_{v,t}},
\qquad
w_{v,t}=\frac{\pi^{\,\mathrm{cue}}_{v,t}}{e^{\zeta}\pi^{\,\mathrm{cue}}_{f,t}+\pi^{\,\mathrm{cue}}_{v,t}}.
$$

**Combined left-belief:**
$$
b^{L}_{t} \;=\; w_{f,t}\,\hat b^{L}_{f,t} + w_{v,t}\,\hat b^{L}_{v,t},
\qquad
v_t \;=\; 2\,b^{L}_{t}-1.
$$

**Volatility-modulated inverse temperature** (symmetric across cue streams):
$$
\beta^{\mathrm{eff}}_{t} \;=\; \beta\,\exp\!\bigl(-\hat\mu_{3,f,t}-\hat\mu_{3,v,t}\bigr).
$$

**Choice likelihood:**
$$
\Pr(y_t=1\mid \boldsymbol\theta) \;=\; s\!\bigl(\beta^{\mathrm{eff}}_{t}\,v_t\bigr).
$$

Card values are deliberately excluded: participants commit to a choice before card values are revealed, so card information cannot enter $v_t$.

---

## 4. Inference, model evidence, and selection

### 4.1 Per-subject inference

For each participant $i$ and each candidate model $m$, parameters are estimated by variational Bayes with a Laplace approximation around the MAP estimate. With prior $p(\boldsymbol\theta\mid m)$ (mostly Gaussian in transformed space; $\kappa,\vartheta\in[0,1]$ via logit transform), the negative free energy

$$
F_{i,m} \;=\; \mathbb{E}_{q}\!\bigl[\log p(\mathbf{y}_i\mid \boldsymbol\theta,m)\bigr] - \mathrm{KL}\!\bigl[q(\boldsymbol\theta)\,\Vert\,p(\boldsymbol\theta\mid m)\bigr]
$$

provides a lower bound on $\log p(\mathbf{y}_i\mid m)$ and the per-subject log-model-evidence input to BMS.

### 4.2 Bayesian Model Selection (Stephan et al., 2009)

With matrix $\mathbf{F}\in\mathbb{R}^{N\times M}$ (`MLTM_extractLME_new`), `spm_BMS` returns model posterior expectations $\langle r_m\rangle$ and exceedance probabilities

$$
\varphi_m \;=\; \Pr\!\bigl(r_m > r_{m'}\;\forall m'\neq m\;\big|\;\mathbf{F}\bigr).
$$

Decisive support is taken as $\varphi_m>0.95$.

> **Note on numerical reproducibility.** `spm_BMS` estimates the exceedance probabilities $\varphi_m$ by Gibbs sampling from the Dirichlet posterior over $r_m$. Because no random seed is set, **each run produces slightly different $\varphi_m$ values** — typically within ≤1 percentage point of one another (e.g., the winning architecture's $\varphi$ has been observed at .966 on one run and .973 on another). The model posterior expectations $\langle r_m\rangle$ are computed analytically and are stable across runs. When comparing your re-run against the values reported in the manuscript, expect this small Gibbs-sampling fluctuation; it does not change the model ordering or the qualitative conclusions.

### 4.3 Identifiability checks

- **Posterior parameter correlations.** For each subject, the Laplace correlation matrix $\mathbf{C}_i=\mathrm{diag}(\mathbf{H}_i^{-1})^{-1/2}\mathbf{H}_i^{-1}\mathrm{diag}(\mathbf{H}_i^{-1})^{-1/2}$ (where $\mathbf{H}_i$ is the Hessian at the MAP) is averaged in Fisher-z space. Used to flag pairwise unidentifiability (e.g.\ $|\mathrm{corr}(\kappa,\vartheta)|>0.8$ would indicate a ridge).
- **Simulation–recovery.** For each subject: simulate responses from the MAP $\hat{\boldsymbol\theta}_i$ on the original input sequence, refit, and compare the recovered to the true parameters via Pearson $r$ and $R^2$.

---

## 5. File ↔ model component mapping

### 5.1 Data cleaning (Python)

| File | Implements |
| --- | --- |
| `data_cleaning/comprehensive_cleaning_pipeline.py` | Reads raw Gorilla CSVs → trial-level `*_cleaned.csv` and HGF-ready `*_hgf_input.csv` (columns $u_{f,t},u_{v,t},d_{f,t},d_{v,t}$, card values, $g_t$). Scores AQ50, DASS21, demographics; assigns $\{ASD, NT\}$. |
| `data_cleaning/data_cleaning_pipeline.py` | Single-file equivalent (also imported as a module by `run_all_cleaning.py`). |
| `data_cleaning/run_all_cleaning.py` | Batch driver over the participant list in `data/participant_metadata.csv`. |

### 5.2 Perceptual model (MATLAB)

| File | Implements |
| --- | --- |
| `Perceptual_Models/hgf_binary3l_facial_verbal.m` | Architecture A (unified facial HGF). |
| `Perceptual_Models/hgf_binary3l_facial_verbal_archB.m` | Architecture B (3 separated facial HGFs sharing $\kappa_f,\omega_f,\vartheta_f$). |
| `Perceptual_Models/hgf_binary3l_facial_verbal_archC.m` | Architecture C (shared $x_{3,f}$, separate $x^{(g)}_{2,f}$). |
| `Perceptual_Models/hgf_binary3l_facial_verbal_gamma.m` | M4 (Architecture A + free $\gamma$). |
| `*_config.m`, `*_transp.m`, `*_namep.m` | Priors $p(\boldsymbol\theta\mid m)$, parameter transforms, parameter labels for each of the above. |
| `*_I1mean_config.m`, `*_I1precision_config.m`, `*_I1both_config.m` | I1 variant prior specifications (free initial $\mu_{2,c,0}$ and/or $\sigma^2_{2,c,0}$). |

### 5.3 Response model (MATLAB)

| File | Implements |
| --- | --- |
| `Response_Models/softmax_facial_verbal.m` | $\Pr(y_t=1\mid\boldsymbol\theta)=s(\beta^{\mathrm{eff}}_{t}\,v_t)$ with precision-weighted $b^{L}_{t}$. |
| `Response_Models/softmax_facial_verbal_config.m` | Prior on $(\zeta,\beta)$ in log space. |
| `softmax_facial_verbal_transp.m`, `_namep.m` | Native-space transform, parameter labels. |

### 5.4 Estimation toolbox (TAPAS, third-party, GPL v3)

| File | Implements |
| --- | --- |
| `Inversion/fitModel.m` | Variational-Bayes inversion: posterior mean, $\mathbf{H}$, $F$. |
| `Optimization/quasinewton_optim.m` | Quasi-Newton (BFGS) MAP optimisation. |
| `Optimization/riddersgradient.m`, `riddershessian.m`, `riddersdiff*.m` | Ridders' polynomial-extrapolation differentiation for $\mathbf{H}$. |
| `Utils/Cov2Corr.m`, `compi_fisherz.m`, `compi_ifisherz.m`, `tapas_logit.m`, `tapas_sgm.m` | Helpers for $\mathbf{C}_i$, Fisher-z, sigmoid/logit. |

### 5.5 Pipeline orchestration (MATLAB)

| File | Implements |
| --- | --- |
| `MLTM_options_new.m` | Paths, model space $\{m_1,\dots,m_4\}$, parameter labels, SPM12 setup. |
| `MLTM_main_new.m` | Wrapper: first-level inversion + second-level analysis. |
| `MLTM_main_I1.m` | Same wrapper restricted to the I1 model space. |
| `MLTM_load_gorilla.m` | Trial-level CSV → $(\mathbf{u}_i,\mathbf{y}_i,\text{metadata})$ for `fitModel`. |
| `MLTM_invert_subject_new.m` | Loop: fit every $(m_{\mathrm{prc}},m_{\mathrm{obs}})$ pair to every subject. |
| `MLTM_extractLME_new.m` | Build $\mathbf{F}\in\mathbb{R}^{N\times M}$. |
| `MLTM_model_selection_new.m` | `spm_BMS(F)` → $\langle r_m\rangle,\varphi_m$, log-Bayes factor plots. |
| `MLTM_check_correlations_new.m` | Average Laplace correlation matrix (Fisher-z mean). |
| `MLTM_extract_parameters_new.m` | Save MAP parameter table + AQ-on-parameters GLM. |
| `MLTM_load_parameters_new.m`, `MLTM_load_zeta_new.m`, `MLTM_load_groups.m` | Per-subject MAP and group-label loaders. |
| `MLTM_second_level_new.m` | Sequencer for the four group-level steps. |
| `MLTM_group_comparison.m` | Parameter-level $2\times 2$ Group $\times$ Condition ANOVA (Type III SS), $t$-tests, Cohen's $d$, partial correlations $r(\mathrm{AQ},\hat\theta_p\mid \mathrm{Cond})$ — **produces Table 4 of the report.** |
| `MLTM_parameter_recovery.m` | Simulation–recovery on the winning model. |
| `MLTM_laplace_kappa_theta.m` | Per-subject $\mathrm{corr}(\kappa,\vartheta)$ from $\mathbf{C}_i$ — flat-likelihood vs. ridge diagnostic. |
| `MLTM_I1_analysis.m` | Group comparison and AQ correlations on $\hat\mu_{2,c,0},\hat\sigma^2_{2,c,0}$ from `I1_Both`. |

### 5.6 Group-level statistics (MATLAB + Python)

| File | Implements |
| --- | --- |
| `analysis/MLTM_sample_characterisation.m` | Table 1 (descriptives, Welch's $t$, Cohen's $d$). |
| `analysis/MLTM_demographics.m` | Demographic equivalence (age, FI, gender) + clinical comparisons (AQ50 subscales, DASS21) + behavioural Group $\times$ Condition. |
| `analysis/MLTM_stratified_AQ_analysis.m` | Condition-stratified zero-order $r$(AQ, parameter) — **produces Table 5 of the report.** |
| `analysis/MLTM_sensitivity_analysis.m` | Re-run after $|z|>2.5$ exclusion — including outlier identification, used in the report sensitivity analysis. |

**Note on canonical sources.** All statistical analyses cited in the report (Tables 1–5, BMS, parameter recovery, Laplace $\kappa\!\leftrightarrow\!\vartheta$, stratified AQ correlations, sensitivity analysis) are produced by the **MATLAB** scripts. Python is used only for the data-cleaning pipeline.

---

## 6. Run order and expected outputs

All commands assume the working directory is the repository root (`4797_model/`).

| # | Command | Produces | Corresponds to |
| --- | --- | --- | --- |
| 1 | `python data_cleaning/comprehensive_cleaning_pipeline.py` | `data/cleaned/<pid>_cleaned.csv`, `data/cleaned/<pid>_hgf_input.csv`, `data/participant_metadata.csv`, `data/quality_report.csv`, `data/cleaning_report.txt`, `data/pid_lookup_PRIVATE.csv` (gitignored) | trial-level $(\mathbf{u}_i,\mathbf{y}_i,g_t,d_{c,t},\dots)$ + $(\mathrm{AQ},\mathrm{DASS},\dots)$ |
| 2 | `MLTM_main_new` (MATLAB, in `hgf_model/`) | `hgf_model/results/<pid>_<m_prc>_<m_obs>.mat` (one per subject × model), `models_F_values.mat`, `BMS_results.mat`, `MLTM_MAP_estimates.csv`, `BMS_*.png`, `parameter_correlations.png` | $\hat{\boldsymbol\theta}_i^{(m)}$, $F_{i,m}$, $\langle r_m\rangle,\varphi_m$, identifiability check |
| 3 | `MLTM_main_I1` (MATLAB) | `hgf_model/results_I1/<pid>_<I1variant>_<m_obs>.mat`, BMS over $\{\text{ArchA}, \text{I1\_Mean}, \text{I1\_Precision}, \text{I1\_Both}\}$ | initial-condition variant comparison |
| 4 | `MLTM_I1_analysis` (MATLAB, after step 3) | `hgf_model/results_I1/I1_Both_MAP_estimates.csv`, group/AQ comparison report | post-hoc on $(\hat\mu_{2,c,0},\hat\sigma^2_{2,c,0})$ |
| 5 | `MLTM_parameter_recovery` (MATLAB, after step 2) | `hgf_model/results/parameter_recovery_results.mat`, `_report.txt`, `_scatter.png` | recovery $r_{\hat\theta\to\tilde\theta}$ for each parameter |
| 6 | `MLTM_laplace_kappa_theta(MLTM_options_new())` (MATLAB) | `hgf_model/results/laplace_kappa_theta.csv`, `_hist.png`, `.txt` | per-subject $\mathrm{corr}(\kappa_c,\vartheta_c)$ flat-vs-ridge diagnostic |
| 7 | `MLTM_sample_characterisation` (MATLAB, in `analysis/`) | Table 1 figure + console paragraph | sample descriptives |
| 8a | `MLTM_demographics` (MATLAB, in `analysis/`) | demographic and clinical comparison console output (no figures) | Table 1 territory (demographic equivalence + AQ subscales + DASS21) |
| 8b | `MLTM_group_comparison(MLTM_options_new())` (MATLAB, in `hgf_model/`; **also runs automatically as the final step of `MLTM_main_new`**) | parameter ANOVA console output + boxplots | **Table 4** ($2\times 2$ Group $\times$ Condition on $\hat\theta$) |
| 9 | `MLTM_stratified_AQ_analysis` (MATLAB) | `hgf_model/results_stratified/*.csv` | condition-stratified $r(\mathrm{AQ},\hat\theta)$ on Arch A and I1\_Both |
| 10 | `MLTM_sensitivity_analysis` (MATLAB) | `hgf_model/results/sensitivity_analysis_report.txt` | full-vs-reduced sample contrasts |
The minimal path through the report is **steps 1, 2, 5, 6, 7, 8a, 8b, 9, 10**; steps 3–4 are needed for the I1 supplementary analysis. The task-design figure (Figure 1 in the report) was generated separately by a one-off MATLAB utility that does not ship with the repo.

---

## 7. Mapping outputs to the report

| Report element | Produced by | Located in |
| --- | --- | --- |
| Table 1 (descriptives) | step 7 | `data/figures/` |
| Figure 1 (task design) | static — produced by a one-off utility not included in the repo | report PDF only |
| Figure 2 (trial structure) | hand-drawn / external | — |
| Table 2 (BMS) | step 2 (`BMS_results.mat`) | `hgf_model/results/` |
| Table 3 (MAP estimates) | step 2 | `MLTM_MAP_estimates.csv` |
| Table 4 (Group × Condition ANOVA) | step 8b | `hgf_model/results/` and console |
| Table 5 (AQ partial correlations, stratified) | step 9 | `hgf_model/results_stratified/` |
| Figure 3 (dimensional + categorical) | step 8b / step 9 | `data/figures/` |
| I1-variant analysis | steps 3–4 | `hgf_model/results_I1/` |
| Parameter recovery | step 5 | `hgf_model/results/` |
| Laplace $\kappa$–$\vartheta$ correlations | step 6 | `hgf_model/results/` |
| Sensitivity analysis | step 10 | `hgf_model/results/sensitivity_analysis_report.txt` |
