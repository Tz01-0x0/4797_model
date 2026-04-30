# Bundled SPM subset — `spm_BMS` and its dependencies

This folder is **not** a copy of the full SPM12 toolbox. It contains only the
files needed to run Bayesian Model Selection (`spm_BMS`) over a matrix of
per-subject log-model evidences:

| File | Purpose |
| --- | --- |
| `spm_BMS.m` | top-level BMS function (Stephan et al., 2009; Rigoux et al., 2014) |
| `spm_BMS_F.m`, `spm_BMS_F_smpl.m` | free-energy and sampling helpers |
| `spm_BMS_bor.m` | Bayes Omnibus Risk |
| `spm_Bcdf.m`, `spm_Dpdf.m` | beta CDF, Dirichlet PDF |
| `spm_dirichlet_exceedance.m` | exceedance probabilities by sampling |
| `spm_gamrnd.m` | help / fallback stub for the compiled Gamma sampler |
| `spm_gamrnd.mexa64` / `.mexmaci64` / `.mexmaca64` / `.mexw64` | platform-specific compiled MEX binaries for `spm_gamrnd` (Linux x86_64, macOS Intel, macOS Apple Silicon, Windows x86_64) |

The eight `.m` files form a closed dependency graph: they call each other but
nothing outside this folder (the `plot/axes` calls inside `spm_BMS.m` are
guarded by `do_plot=1`, which the project's MATLAB pipeline never sets).
`spm_gamrnd.m` is just an SPM12-supplied help-only stub — at runtime MATLAB
automatically picks the matching `spm_gamrnd.mex*` binary for the host
platform, so no compilation is required.

## Source and licence

All files are taken verbatim from SPM12
(<https://www.fil.ion.ucl.ac.uk/spm/>), copyright © 2008–2022 Wellcome
Centre for Human Neuroimaging, redistributed under the GNU General Public
Licence v2 (see `LICENCE`). Nothing in this folder has been modified
relative to the upstream SPM12 source.

If you need the full SPM12 toolbox for any other purpose (preprocessing,
GLM, DCM, …) download it directly from the SPM website. For the analyses
in this repository, this bundled subset is sufficient — no further
installation is required.
