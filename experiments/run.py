#!/usr/bin/env python3
"""Run numerical and small neural studies, or an explicit GPU run plan."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Study:
    name: str
    suite: str
    device: bool = False
    output_argument: bool = False


STUDIES = (
    Study('exact_joint_gradient_interior', 'numerical'),
    Study('timescale_closure', 'numerical'),
    Study('face_energy_barrier', 'numerical'),
    Study('sign_changing_mobility_budget', 'numerical'),
    Study('nonquadratic_pl_interior', 'numerical'),
    Study('shared_bias_invariance', 'numerical'),
    Study('multiexpert_face_mobility', 'numerical'),
    Study('feature_escape_slope', 'numerical'),
    Study('face_basin_map', 'numerical'),
    Study('mobility_exponent_continuum', 'numerical', output_argument=True),
    Study('input_dependent_break_freeze', 'neural', device=True),
    Study('shared_global_load_balancing', 'neural', device=True),
    Study('shared_global_topk', 'neural', device=True),
    Study('neural_capacity_break_freeze', 'neural', device=True),
    Study('architecture_level_moe_check', 'neural', device=True),
    Study('transformer_moe_agnews', 'neural', device=True),
    Study('agnews_compute_matched_dense', 'neural', device=True),
    Study('digits_moe_validation', 'neural', device=True),
    Study('uci_har_moe_validation', 'neural', device=True),
    Study('multidomain_small_moe', 'protocol'),
    Study('multidomain_controlled_moe', 'protocol'),
    Study('multidomain_capacity_robustness', 'protocol'),
    Study('agnews_structure_comparison', 'protocol'),
    Study('agnews_no_drop_clamp_release', 'protocol'),
    Study('agnews_capacity_clamp_release', 'protocol'),
    Study('router_update_no_drop', 'protocol'),
    Study('router_update_capacity', 'protocol'),
)
BY_NAME = {study.name: study for study in STUDIES}


def command(study: Study, python: str, args: list[str]) -> list[str]:
    return [python, '-m', f'experiments.studies.{study.name}', *args]


def plan_commands(path: Path, python: str) -> list[tuple[str, list[str]]]:
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict) or plan.get('version') != 1:
        raise ValueError('Run plan must be an object with version=1')
    runs = plan.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('Run plan must contain a non-empty runs list')
    result = []
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict) or run.get('experiment') not in BY_NAME:
            raise ValueError(f'Unknown experiment in run {index}')
        args = run.get('args')
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError(f'Run {index} args must be a list of strings')
        label = run.get('label', f"{run['experiment']}-{index}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f'Run {index} requires a non-empty label')
        result.append((label, command(BY_NAME[run['experiment']], python, args)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--only', nargs='+', choices=list(BY_NAME))
    selection.add_argument('--run-plan', type=Path)
    parser.add_argument('--suite', choices=['numerical', 'neural', 'all'], default='numerical')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    if len(BY_NAME) != len(STUDIES):
        raise RuntimeError('Duplicate study in registry')
    for study in STUDIES:
        if not (ROOT/'experiments/studies'/f'{study.name}.py').is_file():
            raise FileNotFoundError(study.name)
    if args.list:
        print('name\tsuite\tmodule')
        for study in STUDIES:
            print(f'{study.name}\t{study.suite}\t{study.name}')
        return
    output = args.output_dir.resolve()
    if args.run_plan:
        if args.device != 'cpu' or args.suite != 'numerical':
            parser.error('--run-plan supplies its own per-run device and settings')
        commands = plan_commands(args.run_plan.resolve(), args.python)
    else:
        studies = [BY_NAME[name] for name in dict.fromkeys(args.only)] if args.only else [
            study for study in STUDIES if study.suite != 'protocol' and
            (args.suite == 'all' or study.suite == args.suite)]
        if any(study.suite == 'protocol' for study in studies):
            parser.error('Dataset-dependent studies require --run-plan with explicit arguments')
        commands = []
        for study in studies:
            flags = ['--device', args.device] if study.device else []
            if study.output_argument:
                flags += ['--output-dir', str(output/'tables')]
            commands.append((study.name, command(study, args.python, flags)))
    environment = dict(os.environ, MOE_OUTPUT_DIR=str(output))
    for label, argv in commands:
        print(f'{label}: {shlex.join(argv)}', flush=True)
        if not args.dry_run:
            subprocess.run(argv, cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
