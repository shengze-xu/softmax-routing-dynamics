#!/usr/bin/env python3
"""Restricted mean time to confirmed recovery through 1,024 steps.

Confirmation requires two consecutive qualifying evaluations. Reconstruct
confirmation times from encoded onset means and recovery fractions, then
summarize paired effects by seed."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from experiments.analysis.analyze_recovery_endpoint_sensitivity import SPECS, sign_p

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'outputs/tables'
TAU, GRID, OLD_CODE, CELLS = 1024, 64, 1088, 16


def confirmed_rmst(score: float, fraction: float) -> float:
    if not 0 <= fraction <= 1 or not np.isfinite(score):
        raise ValueError('Invalid score or event fraction')
    if abs(CELLS*fraction-round(CELLS*fraction)) > 1e-9:
        raise ValueError('Fraction must represent 16 complete cells')
    event_sum = CELLS*(score-OLD_CODE*(1-fraction))
    count = round(CELLS*fraction)
    if not GRID*count-1e-8 <= event_sum <= (TAU-GRID)*count+1e-8:
        raise ValueError('Onset sum is incompatible with observed evaluation times')
    if abs(event_sum/GRID-round(event_sum/GRID)) > 1e-8:
        raise ValueError('Onset sum is off the evaluation grid')
    return score-(OLD_CODE-TAU)*(1-fraction)+GRID*fraction


def interval(values, draws=100000, seed=20260831):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means,[.025,.975]))


def analyze(tables=TABLES):
    seeds, summaries=[],[]
    for cohort, filename, expected, fkey, pkey, ffkey, pfkey in SPECS:
        path=tables/filename
        with path.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        if sorted(int(r['seed']) for r in rows)!=sorted(expected):
            raise ValueError('Unexpected seed inventory: '+filename)
        for row in rows:
            control=confirmed_rmst(float(row[fkey]),float(row[ffkey]))
            projected=confirmed_rmst(float(row[pkey]),float(row[pfkey]))
            seeds.append(dict(cohort=cohort,seed=int(row['seed']),cells_per_condition=CELLS,
                restriction_steps=TAU,control_rmst=control,projected_rmst=projected,
                projected_minus_control_rmst=projected-control,
                control_confirmed_fraction=float(row[ffkey]),projected_confirmed_fraction=float(row[pfkey])))
        group=[s for s in seeds if s['cohort']==cohort]
        diff=[s['projected_minus_control_rmst'] for s in group]
        lo,hi=interval(diff)
        summaries.append(dict(cohort=cohort,seed_count=len(group),restriction_steps=TAU,
            control_rmst=float(np.mean([s['control_rmst'] for s in group])),
            projected_rmst=float(np.mean([s['projected_rmst'] for s in group])),
            projected_minus_control_rmst=float(np.mean(diff)),bootstrap_95_low=lo,bootstrap_95_high=hi,
            positive_pairs=sum(d>0 for d in diff),negative_pairs=sum(d<0 for d in diff),ties=sum(d==0 for d in diff),
            exact_two_sided_sign_p=sign_p(diff)))
    return seeds,summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/confirmed_recovery')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    seeds,summaries=analyze(args.input_dir.resolve())
    for stem,rows in [('per_seed',seeds),('summary',summaries)]:
        with (args.output_dir/f'confirmed_recovery_rmst_{stem}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summaries, indent=2))

if __name__=='__main__':main()
