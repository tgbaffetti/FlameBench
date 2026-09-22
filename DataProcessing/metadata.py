"""Explicit dataset metadata; paths are relative to its "data_root" (itself
relative to the metadata file; defaults to the metadata's own directory)."""
import json
from pathlib import Path

FIELDS = ["p", "U1", "U3", "rho", "T", "mix:Q", "CH4", "O2", "H2O", "CO2", "OH"]
UNITS = ["Pa", "m/s", "m/s", "kg/m^3", "K", "W/m^3"] + ["1"] * 5


def load_metadata(path):
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    if result["dt"] <= 0 or len(set(result["fields"])) != len(result["fields"]):
        raise ValueError("Invalid dt or duplicate fields")
    root = (path.parent / result.get("data_root", ".")).resolve()
    result["data_root"] = str(root)
    names = set()
    for case in result["cases"]:
        if case["name"] in names or case["split"] not in {"training", "test"}:
            raise ValueError("Duplicate case name or invalid split")
        names.add(case["name"])
        for key in ("data", "phi"):
            case[key] = str((root / case[key]).resolve())
    for key in ("cell_volumes", "coordinates", "grid_indices"):
        if result.get(key):
            result[key] = str((root / result[key]).resolve())
    return result
