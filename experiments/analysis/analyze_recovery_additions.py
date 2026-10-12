"""Threshold sensitivity and all-complement fixed-token trajectory summaries.

Usage: python3 -m experiments.analysis.analyze_recovery_additions
Extraction: experiments/analysis/extract_recovery_additions.py on the data host.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from experiments.analysis.analyze_recovery_endpoint_sensitivity import sign_p
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / 'outputs/recovery_input'
BRANCHES = ['free_release', 'zero_advantage_gradient_projection_reference']

def interval(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(20260831)
    means = x[rng.integers(0, len(x), (100000, len(x)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/recovery_analysis')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.input_dir / 'extracted.json').read_text())
    original = json.loads((args.input_dir / 'inventory.json').read_text())
    known = {p['path']: p['sha256'] for g in original['groups'].values() for p in g['files']}
    for source in data['sources']:
        if source['path'].endswith('result.json'):
            assert source['sha256'] == known[source['path']]
    cells = pd.DataFrame(data['threshold_cells'], columns=['group','seed','initial_mass','branch','layer','threshold','time','confirmed'])
    cells['cohort'] = np.where(cells.group.str.startswith('no_drop'), 'no_drop', 'capacity')
    per = cells.groupby(['cohort','seed','threshold','branch'])[['time','confirmed']].mean().reset_index()
    per.to_csv(out / 'threshold_per_seed.csv', index=False)
    rows = []
    for (cohort, theta), frame in per.groupby(['cohort','threshold']):
        pivot = frame.pivot(index='seed', columns='branch', values='time')
        d = pivot[BRANCHES[1]] - pivot[BRANCHES[0]]
        lo, hi = interval(d)
        event = frame.groupby('branch').confirmed.mean()
        rows.append(dict(cohort=cohort,threshold=theta,seeds=len(d),control=pivot[BRANCHES[0]].mean(),projection=pivot[BRANCHES[1]].mean(),difference=d.mean(),ci_low=lo,ci_high=hi,positive=int((d>0).sum()),negative=int((d<0).sum()),ties=int((d==0).sum()),control_confirmed=event[BRANCHES[0]],projection_confirmed=event[BRANCHES[1]],exact_two_sided_sign_p=sign_p(d.tolist())))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'threshold_summary.csv', index=False)
    # The 0.05 threshold with second-evaluation confirmation is the main RMST endpoint.
    summary[summary.threshold==0.05].to_csv(out / 'confirmed_recovery_rmst_summary.csv', index=False)
    p = pd.DataFrame(data['probe_trajectories'],columns=['cohort','seed','initial_mass','branch','step','count','A','U','C','budget','margin','above','probability'])
    for c in ['A','U','C','budget','margin','above','probability']:
        p[c] /= p['count']
    # Equal mass weights within a seed; token weights within each mass.
    means = p.groupby(['cohort','seed','branch','step'])[['A','U','C','budget','margin','above','probability']].mean().reset_index()
    means.to_csv(out / 'logodds_per_seed.csv', index=False)
    results=[]
    for cohort, f in means[means.step==1024].groupby('cohort'):
        row={'cohort':cohort}
        for metric in ['A','above','margin']:
            pivot=f.pivot(index='seed',columns='branch',values=metric)
            delta=pivot[BRANCHES[1]]-pivot[BRANCHES[0]]
            row[metric]={'control':pivot[BRANCHES[0]].mean(),'projection':pivot[BRANCHES[1]].mean(),'difference':delta.mean(),'ci':interval(delta),'positive':int((delta>0).sum()),'negative':int((delta<0).sum())}
        results.append(row)
    report={'integrity':data['integrity'],'thresholds':rows,'terminal_probes':results,'initially_subthreshold_counts':p[p.step==64].groupby(['cohort','branch'])['count'].sum().to_dict()}
    report['initially_subthreshold_counts']={str(k):int(v) for k,v in report['initially_subthreshold_counts'].items()}
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
