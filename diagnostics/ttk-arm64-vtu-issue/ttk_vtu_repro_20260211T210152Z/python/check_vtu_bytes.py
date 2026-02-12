#!/usr/bin/env python3
import sys
from pathlib import Path

def hexdump(b: bytes) -> str:
    return " ".join(f"{x:02x}" for x in b)

def main():
    if len(sys.argv) != 2:
        print("usage: check_vtu_bytes.py file.vtu", file=sys.stderr)
        sys.exit(2)
    p = Path(sys.argv[1])
    data = p.read_bytes()

    print("size:", len(data))
    early = data[:4096]
    nul_count = early.count(b"\x00")
    print("NULs in first 4096 bytes:", nul_count)

    idx = 2495
    lo = max(0, idx - 64)
    hi = min(len(data), idx + 64)
    chunk = data[lo:hi]
    print(f"hexdump around {idx} (bytes {lo}:{hi}):")
    print(hexdump(chunk))

    # Show the same region as printable text (replace non-printables)
    txt = "".join(chr(c) if 32 <= c <= 126 else "." for c in chunk)
    print("ascii view:")
    print(txt)

if __name__ == "__main__":
    main()
