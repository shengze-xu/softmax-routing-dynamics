"""Disjoint class-balanced validation sets for selection and recovery.

Selection retains the original per-class prefix. Recovery uses the subsequent
rows, with original dataset row IDs recorded for reproducibility.
"""
import hashlib
import numpy as np
import torch


def validation_sets(dataset, count):
    domains = np.asarray(dataset.arrays['validation_domains'])
    classes = len(dataset.domain_names)
    if count < classes:
        raise ValueError('Each validation set must retain every class')
    per_class, remainder = divmod(count, classes)
    selection, recovery = [], []
    for group in range(classes):
        candidates = np.flatnonzero(domains == group)
        take = per_class + int(group < remainder)
        if len(candidates) < 2*take:
            raise ValueError(f'Class {group} needs {2*take} validation rows')
        selection.extend(candidates[:take].tolist())
        recovery.extend(candidates[take:2*take].tolist())
    selection=np.array(sorted(selection),dtype=np.int64)
    recovery=np.array(sorted(recovery),dtype=np.int64)
    assert np.intersect1d(selection,recovery).size == 0
    def materialize(indices):
        return tuple(torch.from_numpy(np.asarray(dataset.arrays['validation_'+field][indices],dtype=np.int64))
                     for field in ('tokens','lengths','domains'))
    metadata=dict(design='disjoint_selection_and_recovery_v1',
        selection_count=count, recovery_count=count, overlap_count=0,
        selection_row_ids=selection.tolist(), recovery_row_ids=recovery.tolist(),
        selection_index_sha256=hashlib.sha256(selection.astype('<i8').tobytes()).hexdigest(),
        recovery_index_sha256=hashlib.sha256(recovery.astype('<i8').tobytes()).hexdigest())
    return materialize(selection),materialize(recovery),metadata
