"""Read-only extraction from original recovery results and fixed-token probes.

Read run directories specified by --config. Write the extracted numerical
records to --output-dir. Cohort mappings are explicit to keep capacity reruns separate.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

BRANCHES = ['free_release', 'zero_advantage_gradient_projection_reference']
THRESHOLDS = [.04, .05, .06, .08]

def logit(p):
    assert 0 < p < 1
    return math.log(p) - math.log1p(-p)

def run(root, specifications):
    root = Path(root).resolve()
    allowed = {'no_drop_primary','no_drop_additional','capacity_primary','capacity_probe_rerun'}
    if not specifications or set(specifications)-allowed:
        raise ValueError('Unknown or empty recovery groups')
    groups = {name:[(value['directory'], seed) for seed in value['seeds']] for name,value in specifications.items()}
    result = {'threshold_cells': [], 'probe_trajectories': [], 'integrity': {'probe_rows': 0, 'max_probability_continuity_error': 0., 'max_logit_continuity_error': 0., 'max_telescoping_error': 0., 'max_pair_initial_error': 0.}}
    for group, runs in groups.items():
        for folder, seed in runs:
            directory = root / folder / str(seed)
            path = directory / 'result.json'
            data = path.read_bytes()
            obj = json.loads(data)
            del data
            if group != 'capacity_probe_rerun':
                for mass, cell in obj['cells'].items():
                    branches = cell['branches']
                    for branch in BRANCHES:
                        b = branches[branch]
                        steps = b['evaluation_steps']
                        assert steps == list(range(64, 1025, 64))
                        for layer in [1, 2, 3, 4]:
                            probs = [e['raw_target_probability_by_layer'][layer] for e in b['evaluations']]
                            assert len(probs) == 16 and all(0 <= p <= 1 for p in probs)
                            for theta in THRESHOLDS:
                                events = [steps[k] for k in range(1, 16) if probs[k-1] >= theta and probs[k] >= theta]
                                result['threshold_cells'].append([group, seed, float(mass), branch, layer, theta, events[0] if events else 1024, bool(events)])
            del obj
            if group not in ['no_drop_additional', 'capacity_probe_rerun']:
                continue
            for mass in [.005, .01, .02, .04]:
                paired_initial = {}
                for branch in BRANCHES:
                    path = directory / ('probe_decomposition_mass_%.3f_%s.csv.gz' % (mass, branch))
                    tracks, sums = {}, {}
                    with gzip.open(path, 'rt') as f:
                        for row in csv.DictReader(f):
                            layer, token = int(row['layer_zero_based']), int(row['probe_token_id'])
                            start, end = int(row['step_start']), int(row['step_end'])
                            p, pnext = float(row['target_probability_before']), float(row['target_probability_after'])
                            q, qnext = float(row['target_logit_before']), float(row['target_logit_after'])
                            key = (layer, token)
                            if key not in tracks:
                                assert start == 0
                                tracks[key] = [p, q, p, q, 0, 0.]
                                if branch == BRANCHES[0]:
                                    paired_initial[key] = (p, q)
                                else:
                                    e = max(abs(p-paired_initial[key][0]), abs(q-paired_initial[key][1]))
                                    result['integrity']['max_pair_initial_error'] = max(result['integrity']['max_pair_initial_error'], e)
                                    assert e < 1e-12
                            t = tracks[key]
                            assert t[4] == start
                            for name, error in [('max_probability_continuity_error', abs(t[2]-p)), ('max_logit_continuity_error', abs(t[3]-q))]:
                                result['integrity'][name] = max(result['integrity'][name], error)
                                assert error < 1e-12
                            increment = logit(pnext)-logit(p)
                            t[5] += increment
                            A, U = logit(pnext)-logit(t[0]), qnext-t[1]
                            budget = logit(.05)-logit(t[0])
                            result['integrity']['max_telescoping_error'] = max(result['integrity']['max_telescoping_error'], abs(A-t[5]))
                            # Fixed initially subthreshold subset; membership is shared by both branches.
                            if t[0] < .05:
                                agg = sums.setdefault(end, [0]*8)
                                values = [1, A, U, U-A, budget, A-budget, pnext >= .05, pnext]
                                for i, v in enumerate(values): agg[i] += v
                            t[2:5] = pnext, qnext, end
                            result['integrity']['probe_rows'] += 1
                    assert len(tracks) == 512 and all(t[4] == 1024 for t in tracks.values())
                    for step, values in sorted(sums.items()):
                        result['probe_trajectories'].append([group, seed, mass, branch, step] + values)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    payload=run(args.root,config['groups'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'extracted.json').write_text(json.dumps(payload)+'\n')
    print(json.dumps(payload['integrity'],indent=2))

if __name__ == '__main__':
    main()
