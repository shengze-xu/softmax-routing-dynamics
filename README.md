# Softmax Routing Dynamics in Mixture-of-Experts Models: Collapse and Recovery

Experimental code for the paper's continuous routing dynamics and neural
recovery interventions. This repository contains algorithm implementations,
experiment entry points, data preparation, numerical/statistical analysis,
and checks of the recorded outputs. It produces CSV, JSON, NPZ, and optional
raw-log files. Plotting, manuscript sources, figure assembly, and publication
build scripts remain in the research repository.

## Layout

- `experiments/common/`: simplex dynamics, numerical utilities, validation
  splitting, and exact affine logit decomposition.
- `experiments/models/`: shared model, data, and training implementations used
  by the experiment scripts.
- `experiments/studies/`: the 27 experiment entry points used by the current
  main text and appendices.
- `experiments/data/`: dataset downloads, tokenization, and corpus verification.
- `experiments/analysis/`: paired statistics, recovery endpoints, fixed-token
  log-odds, and additional numerical checks.
- `experiments/audits/`: gradient-intervention and result-integrity checks.
- `experiments/protocols/`: fixed numerical designs, seed inventories, and
  SHA-256 checksums for the direct interventions.
- `configs/`: portable launch plans and recorded neural configurations.
- `experiments/tests/`: CPU checks of numerical identities and intervention
  behavior.

Generated results, datasets, and checkpoints are ignored by Git. Paths in
launch plans are relative to this repository root; replace `data/agnews_prepared`
if your audited prepared dataset is elsewhere. All commands below run from
this root. The local folder can be copied or cloned independently of the
research repository.

## Paths and relocation

The outer repository directory may have any name and may live anywhere.
Repository paths are derived from `__file__`; launch-plan paths are relative
to the repository root. Dataset and run roots are supplied through command-line
arguments or the editable JSON plans.

New corpus manifests store tokenizer and source references relative to the
final dataset directory. Move the enclosing data tree together to preserve
these references. When using an older manifest containing an absolute tokenizer
path, pass `--tokenizer-dir /new/tokenizer/location` to the large multidomain or AG News
corpus auditor. This overrides the location without changing the manifest or
its checksum.

Result auditors accept any recorded GPU name by default and still check CUDA
execution and numerical integrity. Use `--expected-gpu "RTX 5090"` when exact
hardware-name matching is desired. BF16-capable CUDA hardware remains necessary
for the corresponding training experiments.

The raw-result analysis tools use `audit.json` by default, matching the audit
commands below. Use `--audit-name` to select another filename.

The Python package name `experiments`, result schemas such as
`manifest.json` and `result.json`, and the recorded dataset/protocol/branch
identifiers are shared interfaces. They encode the experiment design and should
be kept consistent when changing modules or result formats. They do
not require a particular outer folder, user account, or machine name.

Run metadata can record resolved absolute paths, hostnames, and GPU identities
for provenance. Those recorded values do not replace the paths supplied to
subsequent runs or analyses.

## Environment

The numerical and CPU neural environments were checked with Python 3.11.9.
Create a virtual environment and install the dependencies for your workload:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

- `requirements.txt`: NumPy and SciPy for numerical studies.
- `requirements-neural.txt`: the recorded CPU/small-neural environment,
  including PyTorch and scikit-learn.
- `requirements-cu128.txt`: the recorded RTX 5090/PyTorch 2.8/CUDA 12.8
  environment; install this in a separate GPU environment.
- `requirements-analysis.txt`: the recorded CPU numerical environment plus
  pandas and PyArrow for raw-log and fixed-token analysis. Use a separate CPU
  environment from the NumPy 2.3 GPU environment.
- `requirements-data.txt`: the additional datasets/transformers packages needed
  only when constructing the tokenized corpora. Corpus manifests record source
  revisions, tokenizer metadata, and checksums.

Matplotlib is not required. GPU experiments use CUDA and BF16; the CPU tests
exercise their algorithms on small FP32 tensors, rather than training the
full published models.

## Experiment map

| Paper material | Entry points | Execution |
| --- | --- | --- |
| Face selection and initialization sweep, Sections 3 and 7.2; Appendix B.1 | `face_energy_barrier`, `face_basin_map` | Numerical |
| Joint interior convergence and finite-horizon adaptation, Sections 4 and 7.2; Appendix B.2 | `exact_joint_gradient_interior`, `timescale_closure`, `nonquadratic_pl_interior` | Numerical |
| Conditional recovery scaling, Section 7.3 and Appendix B.3--B.4 | `multiexpert_face_mobility`, `mobility_exponent_continuum`, `sign_changing_mobility_budget`; `analysis.check_extended_recovery_laws` | Numerical |
| Affine routing separation and shared-bias invariance, Appendix C.1 | `shared_bias_invariance`, `feature_escape_slope` | Numerical |
| Synthetic routing comparisons, Appendix C.3 | `input_dependent_break_freeze`, `shared_global_load_balancing`, `shared_global_topk`, `neural_capacity_break_freeze`, `architecture_level_moe_check` | Small neural |
| AG News, Digits, UCI HAR, Appendix C.4 | `transformer_moe_agnews`, `agnews_compute_matched_dense`, `digits_moe_validation`, `uci_har_moe_validation` | Small neural |
| Four-domain scale transfer and capacity, Appendix C.5 | `multidomain_small_moe`, `multidomain_controlled_moe`, `multidomain_capacity_robustness` | Prepared corpus + explicit plan |
| Independent AG News structure and clamp release, Appendices C.5 and D.1 | `agnews_structure_comparison`, `agnews_no_drop_clamp_release`, `agnews_capacity_clamp_release` | Prepared corpus + explicit plan |
| Router-gradient intervention, Section 7.4 and Appendices D.2--E | `router_update_no_drop`, `router_update_capacity` | Prepared corpus + explicit plan |

Entry points use descriptive names throughout the filenames, launch plans,
commands, and generated outputs. Shared model modules provide their training
and data implementations.

## Numerical experiments

List the registry or inspect the numerical suite:

```bash
python -m experiments.run --list
python -m experiments.run --dry-run
```

Run all ten numerical studies, or select a subset:

```bash
python -m experiments.run --suite numerical
python -m experiments.run --only exact_joint_gradient_interior shared_bias_invariance feature_escape_slope
```

The default suite is numerical. Numerical outputs go to `outputs/tables/`
and `outputs/*.npz`. Select a different destination with `--output-dir`.
Individual study modules expose numerical arguments; for example:

```bash
python -m experiments.studies.face_basin_map --help
python -m experiments.studies.exact_joint_gradient_interior --seeds 0 --horizon 2 --dts 0.04 0.02
python -m experiments.analysis.check_extended_recovery_laws --output-dir outputs/tables
python -m experiments.analysis.replay_face_trajectories --input outputs/tables/face_basin_map.csv --output-dir outputs/tables
```

`mobility_exponent_continuum` integrates all logits under a prescribed coefficient `Gamma=x^alpha`
and compares event times with independent quadrature. Scores are unbounded
near zero when `alpha<1`; this validates the conditional continuous law,
rather than fitting an exponent to neural training.

## Small neural experiments

```bash
python -m pip install -r requirements-neural.txt
python -m experiments.run --suite neural --device cpu
```

This runs the nine small neural studies with their recorded default seeds
and settings. The AG News dense baseline (`agnews_compute_matched_dense`) reads the `transformer_moe_agnews` summary,
so run `transformer_moe_agnews` first when selecting studies individually. Synthetic studies
need no external data. Digits uses scikit-learn; AG News and UCI HAR download
and verify their source data when the local caches are absent.

Run paired routing comparisons on the regenerated per-seed outputs:

```bash
python -m experiments.analysis.paired_routing_statistics --input-dir outputs/tables --output-dir outputs/paired_statistics
```

## Prepared corpora

The large experiments verify manifests, array checksums, and corpus audits.
Prepare AG News using its full deduplicated corpus and the GPT-2 tokenizer:

```bash
python -m pip install -r requirements-data.txt
python -m experiments.data.download_agnews --output-dir data/ag_news_csv
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2').save_pretrained('data/gpt2-tokenizer')"
python -m experiments.data.prepare_agnews_corpus --source-dir data/ag_news_csv --tokenizer-dir data/gpt2-tokenizer --output-dir data/agnews_prepared
python -m experiments.data.audit_agnews_corpus --dataset-dir data/agnews_prepared --source-dir data/ag_news_csv
```

For the four-domain corpus, use `prepare_multidomain_small_corpus` for the
small model (256-token fragments) and `prepare_multidomain_large_corpus` for
the larger model (512-token fragments), followed by their corresponding
`audit_multidomain_small_corpus` or `audit_multidomain_large_corpus` modules.
The preparation modules pin source revisions and document-level split rules.
Their fixed hexadecimal split keys preserve the published document partitions;
`--split-key` accepts an alternative hexadecimal key. Use `--help` for token
budgets and tokenizer/output paths. Prepared data and
third-party tokenizer files stay in ignored `data/`.

## GPU runs

Large runs are invoked from explicit JSON plans. Each seed has its own output
directory. Inspect a plan before starting it:

```bash
python -m experiments.run --run-plan configs/router_update_smoke.json --dry-run
python -m experiments.run --run-plan configs/router_update_smoke.json
```

The smoke plan exercises both dispatch policies and fixed-token recording at
reduced scale. The primary intervention plan has 16 independent no-drop seeds,
eight capacity seeds, and eight capacity probe reruns:

```bash
python -m experiments.run --run-plan configs/router_update_runs.json --dry-run
python -m experiments.run --run-plan configs/router_update_runs.json
```

The eight capacity probe reruns reuse the original capacity seed values to
collect missing token/feature records. They are analyzed separately from the
primary capacity recovery cohort and are not additional independent seeds.
Run each result-integrity check with the matching role:

```bash
python -m experiments.audits.audit_router_update_no_drop --result outputs/mechanism/no_drop/5201/result.json --role primary --output outputs/mechanism/no_drop/5201/audit.json
python -m experiments.audits.audit_router_update_capacity --result outputs/mechanism/capacity/5401/result.json --role primary --output outputs/mechanism/capacity/5401/audit.json
```

`configs/agnews_structure_release.json` runs the eight-seed independent
structure comparison and both eight-seed clamp-release comparisons.
Its three architecture/training configuration files record the published
settings. The release-plan SHA-256 arguments identify the included numerical
configurations. The router-gradient interventions validate the fixed design
directly from the included protocol.

```bash
python -m experiments.run --run-plan configs/agnews_structure_release.json --dry-run
python -m experiments.run --run-plan configs/agnews_structure_release.json
```

For `multidomain_small_moe`, `multidomain_controlled_moe`, and `multidomain_capacity_robustness`, provide an explicit plan with the chosen
audited corpus path, output path, seed, and model/training arguments shown by
that module's `--help`. Full GPU training requires the prepared corpora and
the recorded compute budget.

## Recovery statistics

After all intervention runs and probe sidecars are available, edit
`configs/recovery_analysis.json` if your run directories differ, then compute
the paper's recovery and token-level numerical summaries:

```bash
python -m pip install -r requirements-analysis.txt
python -m experiments.analysis.extract_recovery_additions --config configs/recovery_analysis.json --root . --output-dir outputs/recovery_input
python -m experiments.analysis.analyze_recovery_additions --input-dir outputs/recovery_input --output-dir outputs/recovery_analysis
```

The outputs include four-threshold sensitivity, paired-seed bootstrap
intervals, the primary 0.05-threshold RMST summary, fixed-token cumulative
log-odds, and threshold membership fractions. Confirmation occurs at the
second of two consecutive qualifying evaluations; times are truncated at
1,024 steps. Statistical resampling keeps each seed's layers and masses
together. Fixed-token statistics use a single evaluation and a fixed
initially subthreshold subset, with the paper's within-seed weighting.

`analyze_router_update_no_drop`, `analyze_router_update_capacity`, and
`analyze_additional_recovery` provide the original raw-log intake and pooled
encoded-score analyses. `analyze_confirmed_recovery` and
`analyze_recovery_endpoint_sensitivity` accept explicitly supplied per-seed
summary directories. None reads manuscript artifact directories.
The original 1,088 non-recovery score is distinguished from the main
1,024-step confirmation-time RMST.

## Checks

Install the neural and analysis requirements to run the complete CPU checks:

```bash
python -m unittest discover -s experiments/tests -t . -p 'test_*.py'
```

The checks cover the full normal-velocity identity, closed-form/quadrature
agreement, gradient projection, capacity conservation and dropping, logit
reconstruction, confirmation timing, paired statistical units, and portable
input paths. A numerical-only environment can run:

```bash
python -m unittest experiments.tests.test_mobility_exponent_continuum
```

## License

Executable code and repository documentation are distributed under the MIT
license in `LICENSE`.
