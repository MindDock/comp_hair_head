#!/usr/bin/env python3
"""Convert FLAME model from chumpy pickle to numpy .npz format.

Patches chumpy at runtime for Python 3.13 / numpy 2.x compatibility,
loads the model, converts all chumpy objects to numpy arrays,
and saves as .npz (no chumpy dependency needed for loading).
"""

import pickle
import numpy as np
import sys
import inspect


def _patch_chumpy():
    """Patch chumpy for Python 3.13 + numpy 2.x compatibility."""
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

    _deprecated_aliases = {
        'bool': np.bool_,
        'int': np.int_,
        'float': np.float64,
        'complex': np.complex128,
        'object': np.object_,
        'str': np.str_,
        'unicode': np.str_,
    }

    def _patched_getattr(name):
        if name in _deprecated_aliases:
            return _deprecated_aliases[name]
        if name == 'nan':
            return np.nan
        if name == 'inf':
            return np.inf
        raise AttributeError(f"module 'numpy' has no attribute {name!r}")

    np.__getattr__ = _patched_getattr


def _to_numpy(obj):
    """Convert chumpy or sparse objects to numpy arrays."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "r"):
        try:
            result = obj.r
            if isinstance(result, np.ndarray):
                return result
            return np.array(result)
        except Exception:
            pass
    if hasattr(obj, "A"):
        try:
            return np.array(obj.A)
        except Exception:
            pass
    if hasattr(obj, "toarray"):
        try:
            return obj.toarray()
        except Exception:
            pass
    return obj


def convert_flame_model(input_path: str, output_path: str):
    """Convert FLAME model pickle to .npz format."""
    _patch_chumpy()

    import chumpy  # noqa: E402

    print(f"Loading FLAME model from {input_path}...")

    with open(input_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    print(f"Keys: {list(data.keys())}")

    arrays = {}
    metadata = {}
    for k, v in data.items():
        converted = _to_numpy(v)
        if isinstance(converted, np.ndarray):
            arrays[k] = converted
            print(f"  {k}: shape={converted.shape}, dtype={converted.dtype}")
        else:
            metadata[k] = str(converted)
            print(f"  {k}: metadata={converted}")

    print(f"\nSaving converted model to {output_path}...")
    arrays["_metadata_keys"] = np.array(list(metadata.keys()))
    arrays["_metadata_values"] = np.array(list(metadata.values()))
    np.savez(output_path, **arrays)

    print("Done! FLAME model converted successfully.")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "assets/flame/generic_model.pkl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "assets/flame/generic_model.npz"
    convert_flame_model(input_path, output_path)
