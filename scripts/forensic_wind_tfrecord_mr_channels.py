#!/usr/bin/env python3
"""Forensic audit of wind MR-HR TFRecord parsing and MR channel content.

Focus:
- inspect raw byte lengths and parsed shapes
- verify parser assumptions against TFRecord keys/types
- test whether MR channels appear duplicated/highly-correlated
- probe a few alternative parsing hypotheses for LR bytes

This does NOT change any model/pipeline code.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "TensorFlow is required to parse TFRecords for this forensic check. "
            f"Import failed: {exc}"
        )
    return tf


@dataclass
class RecordRaw:
    row: int
    index: int
    h_lr: int
    w_lr: int
    h_hr: int
    w_hr: int
    c: int
    bytes_lr: bytes
    bytes_hr: bytes


@dataclass
class ChannelStats:
    equal_exact: bool
    nearly_equal: bool
    corr: float
    mean_abs_diff: float
    max_abs_diff: float


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def channel_stats(arr: np.ndarray, atol: float = 1e-12) -> ChannelStats:
    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f"Expected [H,W,C>=2], got {arr.shape}")
    ch0 = np.asarray(arr[..., 0], dtype=np.float64)
    ch1 = np.asarray(arr[..., 1], dtype=np.float64)
    diff = np.abs(ch0 - ch1)
    return ChannelStats(
        equal_exact=bool(np.array_equal(ch0, ch1)),
        nearly_equal=bool(np.allclose(ch0, ch1, rtol=0.0, atol=atol)),
        corr=_pearson(ch0, ch1),
        mean_abs_diff=float(np.mean(diff)),
        max_abs_diff=float(np.max(diff)),
    )


def parse_raw_records(path: Path, max_records: int) -> List[RecordRaw]:
    tf = _import_tensorflow()
    out: List[RecordRaw] = []
    for row, raw in enumerate(tf.compat.v1.io.tf_record_iterator(str(path))):
        ex = tf.train.Example()
        ex.ParseFromString(raw)
        f = ex.features.feature

        def _get_i(name: str) -> int:
            return int(f[name].int64_list.value[0])

        def _get_b(name: str) -> bytes:
            return bytes(f[name].bytes_list.value[0])

        out.append(
            RecordRaw(
                row=row,
                index=_get_i("index"),
                h_lr=_get_i("h_LR"),
                w_lr=_get_i("w_LR"),
                h_hr=_get_i("h_HR"),
                w_hr=_get_i("w_HR"),
                c=_get_i("c"),
                bytes_lr=_get_b("data_LR"),
                bytes_hr=_get_b("data_HR"),
            )
        )
        if len(out) >= max_records:
            break
    return out


def expected_bytes(h: int, w: int, c: int, dtype: np.dtype) -> int:
    return int(h * w * c * np.dtype(dtype).itemsize)


def try_parse_lr_hypotheses(rec: RecordRaw) -> List[Dict[str, object]]:
    """Try a small set of alternative LR parsing hypotheses and score channel similarity."""
    hyps: List[Tuple[str, Optional[np.dtype], Optional[Tuple[int, ...]], Optional[str]]] = []

    # baseline assumption used by parser
    hyps.append(("float64_HWC", np.float64, (rec.h_lr, rec.w_lr, rec.c), "C"))

    # wrong dtype hypotheses
    hyps.append(("float32_HWC", np.float32, (rec.h_lr, rec.w_lr, rec.c), "C"))
    hyps.append(("int16_HWC", np.int16, (rec.h_lr, rec.w_lr, rec.c), "C"))

    # alternate layout hypothesis: CHW then transpose
    hyps.append(("float64_CHW_to_HWC", np.float64, (rec.c, rec.h_lr, rec.w_lr), "C"))

    # alternate reshape order hypothesis
    hyps.append(("float64_HWC_fortran", np.float64, (rec.h_lr, rec.w_lr, rec.c), "F"))

    # single-channel duplicated hypothesis (first half == second half)
    hyps.append(("float64_single_channel_dup", np.float64, None, None))

    results: List[Dict[str, object]] = []
    b = rec.bytes_lr

    for name, dt, shp, order in hyps:
        row: Dict[str, object] = {"hypothesis": name, "parsable": False}
        try:
            if name == "float64_single_channel_dup":
                arr = np.frombuffer(b, dtype=np.float64)
                n = rec.h_lr * rec.w_lr
                if arr.size == 2 * n:
                    ch0 = arr[:n].reshape(rec.h_lr, rec.w_lr)
                    ch1 = arr[n:].reshape(rec.h_lr, rec.w_lr)
                    x = np.stack([ch0, ch1], axis=-1)
                elif arr.size == n:
                    ch0 = arr.reshape(rec.h_lr, rec.w_lr)
                    x = np.stack([ch0, ch0], axis=-1)
                else:
                    raise ValueError(f"size {arr.size} incompatible with single-channel hypothesis")
            else:
                assert dt is not None and shp is not None and order is not None
                arr = np.frombuffer(b, dtype=dt)
                need = int(np.prod(shp))
                if arr.size != need:
                    raise ValueError(f"size mismatch arr.size={arr.size} need={need}")
                x = np.reshape(arr, shp, order=order)
                if name == "float64_CHW_to_HWC":
                    x = np.transpose(x, (1, 2, 0))

            row["parsable"] = True
            if x.ndim == 3 and x.shape[-1] >= 2:
                st = channel_stats(np.asarray(x, dtype=np.float64))
                row.update(
                    {
                        "shape": str(tuple(int(v) for v in x.shape)),
                        "equal_exact": st.equal_exact,
                        "nearly_equal": st.nearly_equal,
                        "corr": st.corr,
                        "mean_abs_diff": st.mean_abs_diff,
                        "max_abs_diff": st.max_abs_diff,
                    }
                )
            else:
                row.update({"shape": str(tuple(int(v) for v in x.shape)), "note": "parsed but no 2 channels"})
        except Exception as exc:
            row["error"] = str(exc)

        results.append(row)
    return results


def inspect_schema(path: Path, max_keys: int = 50) -> Dict[str, object]:
    tf = _import_tensorflow()
    it = tf.compat.v1.io.tf_record_iterator(str(path))
    try:
        raw = next(it)
    except StopIteration:
        raise SystemExit(f"No records in {path}")

    ex = tf.train.Example()
    ex.ParseFromString(raw)
    feat = ex.features.feature

    keys = sorted(feat.keys())[:max_keys]
    kinds: Dict[str, str] = {}
    for k in keys:
        f = feat[k]
        if len(f.bytes_list.value) > 0:
            kinds[k] = "bytes_list"
        elif len(f.int64_list.value) > 0:
            kinds[k] = "int64_list"
        elif len(f.float_list.value) > 0:
            kinds[k] = "float_list"
        else:
            kinds[k] = "empty"

    required = ["index", "data_LR", "h_LR", "w_LR", "data_HR", "h_HR", "w_HR", "c"]
    missing = [k for k in required if k not in feat]

    return {"keys": keys, "kinds": kinds, "missing_required": missing}


def fmt_float(v: float) -> str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return f"{v:.6g}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Forensic MR-channel audit for wind MR-HR TFRecord parsing/content")
    ap.add_argument("--tfrecord", type=Path, default=Path("example_data_extension_500/wind_MR-HR.tfrecord"))
    ap.add_argument("--max-records", type=int, default=6, help="How many records to inspect")
    ap.add_argument("--atol-nearly-equal", type=float, default=1e-12)
    args = ap.parse_args()

    schema = inspect_schema(args.tfrecord)
    print("=== TFRecord schema check ===")
    print(f"file: {args.tfrecord}")
    print(f"keys: {schema['keys']}")
    print(f"kinds: {schema['kinds']}")
    if schema["missing_required"]:
        print(f"MISSING REQUIRED KEYS: {schema['missing_required']}")
    else:
        print("required keys present: yes")

    records = parse_raw_records(args.tfrecord, max_records=args.max_records)
    if not records:
        raise SystemExit(f"No records found in {args.tfrecord}")

    print("\n=== Per-record forensic summary ===")
    for rec in records:
        print(f"\n-- row={rec.row} index={rec.index} --")
        print(f"raw byte lengths: data_LR={len(rec.bytes_lr)} data_HR={len(rec.bytes_hr)}")
        print(f"declared shapes/channels: LR=({rec.h_lr},{rec.w_lr},{rec.c}) HR=({rec.h_hr},{rec.w_hr},{rec.c})")
        print("dtype assumptions tested: float64 baseline (+ float32/int16 alternatives)")

        exp_lr_f64 = expected_bytes(rec.h_lr, rec.w_lr, rec.c, np.float64)
        exp_hr_f64 = expected_bytes(rec.h_hr, rec.w_hr, rec.c, np.float64)
        print(f"expected bytes if float64: LR={exp_lr_f64} HR={exp_hr_f64}")
        print(
            f"byte-length match float64? LR={len(rec.bytes_lr)==exp_lr_f64} "
            f"HR={len(rec.bytes_hr)==exp_hr_f64}"
        )

        # Baseline parse
        mr = np.frombuffer(rec.bytes_lr, dtype=np.float64).reshape(rec.h_lr, rec.w_lr, rec.c)
        hr = np.frombuffer(rec.bytes_hr, dtype=np.float64).reshape(rec.h_hr, rec.w_hr, rec.c)
        print(f"parsed baseline shapes: MR={mr.shape} HR={hr.shape}")

        mstats = channel_stats(mr, atol=args.atol_nearly_equal)
        print(
            "MR ch0 vs ch1: "
            f"exact_equal={mstats.equal_exact} nearly_equal={mstats.nearly_equal} "
            f"corr={fmt_float(mstats.corr)} mean_abs_diff={fmt_float(mstats.mean_abs_diff)} "
            f"max_abs_diff={fmt_float(mstats.max_abs_diff)}"
        )

        hstats = channel_stats(hr, atol=args.atol_nearly_equal)
        print(
            "HR ch0 vs ch1: "
            f"exact_equal={hstats.equal_exact} nearly_equal={hstats.nearly_equal} "
            f"corr={fmt_float(hstats.corr)} mean_abs_diff={fmt_float(hstats.mean_abs_diff)} "
            f"max_abs_diff={fmt_float(hstats.max_abs_diff)}"
        )

        print("alternative LR parsing hypotheses:")
        hyps = try_parse_lr_hypotheses(rec)
        for h in hyps:
            if h.get("parsable"):
                print(
                    "  - {hyp}: parsable shape={shape} exact={exact} near={near} corr={corr} mean|d|={mad} max|d|={mxd}".format(
                        hyp=h["hypothesis"],
                        shape=h.get("shape", "?"),
                        exact=h.get("equal_exact", "?"),
                        near=h.get("nearly_equal", "?"),
                        corr=fmt_float(float(h.get("corr", float("nan")))),
                        mad=fmt_float(float(h.get("mean_abs_diff", float("nan")))),
                        mxd=fmt_float(float(h.get("max_abs_diff", float("nan")))),
                    )
                )
            else:
                print(f"  - {h['hypothesis']}: not parsable ({h.get('error', 'unknown error')})")

    print("\n=== Short conclusion scaffold ===")
    print("1) Parser-schema consistency: see required keys/kinds and float64 byte-length matches above.")
    print("2) MR channel duplication signal: check exact/nearly-equal/high-correlation and |diff| stats above.")
    print("3) If MR channels look duplicated while HR channels do not, likely issue is in LR content generation, not orientation.")
    print("4) If baseline float64-HWC parses cleanly and alternatives do not improve plausibility, parser is likely correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
