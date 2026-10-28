# Softmax Routing Dynamics in Mixture-of-Experts Models: Collapse and Recovery

Official implementation for the paper by Shengze Xu, Maolin Deng, and Tieyong Zeng.
The code covers continuous routing dynamics, neural routing experiments, and
recovery interventions.

## Installation

Use Python 3.11 and pip 25.1 or later:

```bash
git clone https://github.com/shengze-xu/softmax-routing-dynamics.git
cd softmax-routing-dynamics
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group numerical
```

For neural experiments, analysis, or corpus preparation, install the relevant
dependency groups:

```bash
python -m pip install --group neural --group analysis --group data
```

GPU experiments use the `cu128` group in a separate environment.

## Running experiments

```bash
python -m experiments.run --list
python -m experiments.run --dry-run
python -m experiments.run --suite numerical
python -m experiments.run --suite neural --device cpu
```

Select individual studies with `--only`. Each study also supports `--help`:

```bash
python -m experiments.run --only face_energy_barrier timescale_closure
python -m experiments.studies.router_update_no_drop --help
```

GPU launch plans and model settings are in `configs/`. Set dataset and output
paths before running a plan:

```bash
python -m experiments.run --run-plan configs/router_update_smoke.json --dry-run
```

Data preparation and result analysis are in `experiments/data/` and
`experiments/analysis/`; each command provides its arguments through `--help`.
Generated data, checkpoints, and results are excluded from Git.

## Tests

With the `neural` and `analysis` groups installed:

```bash
python -m unittest discover -s tests
```

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Code and documentation
are distributed under the [MIT license](LICENSE).
