"""Write feature tables. Layer-wise vectors are stored one column per layer
(``mean_l00`` ... ``mean_l12``) as fixed-size float16 lists so a probe can read a
single layer without loading the rest (parquet column pruning)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _vec_col(x: np.ndarray, dtype=np.float16) -> pa.Array:
    x = np.ascontiguousarray(x.astype(dtype))
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


def write_layered(path: Path, segment_ids: list[str], arrays: dict[str, np.ndarray]) -> None:
    """arrays: name -> [N, L, D]; written as columns f"{name}_l{k:02d}"."""
    cols = {"segment_id": pa.array(segment_ids)}
    for name, a in arrays.items():
        for k in range(a.shape[1]):
            cols[f"{name}_l{k:02d}"] = _vec_col(a[:, k])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path, compression="zstd")


def write_vectors(path: Path, segment_ids: list[str], name: str, a: np.ndarray, dtype=np.float16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"segment_id": pa.array(segment_ids), name: _vec_col(a, dtype)}), path,
                   compression="zstd")


def read_layer(path: Path, name: str, layer: int) -> tuple[list[str], np.ndarray]:
    t = pq.read_table(path, columns=["segment_id", f"{name}_l{layer:02d}"])
    col = t.column(1).combine_chunks()
    d = col.type.list_size
    return t.column(0).to_pylist(), col.flatten().to_numpy().reshape(-1, d).astype(np.float32)
