#!/usr/bin/env python3
"""Replay four fixed initializations from the simplex basin sweep."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import argparse
import csv
import numpy as np
from experiments.studies.face_basin_map import rk4_step, row_softmax
from experiments.studies.face_energy_barrier import SCENARIOS


def main():
    root = Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=root/'outputs/tables')
    args=parser.parse_args()
    directory=args.output_dir
    directory.mkdir(parents=True,exist_ok=True)
    with args.input.open() as handle:
        records = list(csv.DictReader(handle))
    ids = (0, 1, 3, 4)
    chosen = [records[i] for i in ids]
    initial = np.array([[float(r[f'initial_p{k}']) for k in (1, 2, 3)] for r in chosen])
    logits = np.log(initial)
    logits -= logits.mean(axis=1, keepdims=True)
    model = SCENARIOS[0]
    dt, horizon = .05, 2000.
    entries = np.full(4, np.nan)
    rows = []
    for step in range(round(horizon / dt) + 1):
        if step:
            logits = rk4_step(logits, model.experts, model.target, dt)
        probability = row_softmax(logits)
        losses = ((probability @ model.experts - model.target) ** 2).sum(axis=1)
        for i, index in enumerate(ids):
            entered = np.isnan(entries[i]) and losses[i] < float(chosen[i]['barrier']) - (1e-10 if step else 0)
            if entered:
                entries[i] = step * dt
            if step % 40 == 0 or entered:
                rows.append((index, step * dt, *probability[i], losses[i], int(entered)))
    terminal_error = max(abs(probability[i, 2] - float(r['terminal_outside_mass'])) for i, r in enumerate(chosen))
    entry_error = max(abs(entries[i] - float(r['entry_time'])) for i, r in enumerate(chosen))
    if terminal_error > 1e-12 or entry_error > 1e-9:
        raise AssertionError((terminal_error, entry_error))
    with (directory / 'representative_trajectories.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('initialization', 'time', 'p1', 'p2', 'p3', 'loss', 'first_sampled_entry'))
        writer.writerows(rows)
    print(f'Replayed {ids}; terminal mass error={terminal_error:.3g}; entry-time error={entry_error:.3g}')


if __name__ == '__main__':
    main()
