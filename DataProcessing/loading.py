"""Shared DataLoader options for notebooks and experiments."""
from torch.utils.data import DataLoader


def make_loader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False,
                persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn"):
    """Spawn CPU workers safely even after CUDA initialization; keep every sample.

    Worker-only options are ignored when num_workers=0. Prefetching holds roughly
    num_workers * prefetch_factor batches, so start small for image windows.
    """
    options = {}
    if num_workers > 0:
        options = dict(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
                       multiprocessing_context=multiprocessing_context)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, **options)
