#!/usr/bin/env python3
import sys, re, base64
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("usage: check_vtu_appended_base64.py file.vtu", file=sys.stderr)
        sys.exit(2)

    s = Path(sys.argv[1]).read_text(errors="strict")

    # Grab content inside <AppendedData ...> ... </AppendedData>
    m = re.search(r"<AppendedData[^>]*>(.*?)</AppendedData>", s, flags=re.S)
    if not m:
        print("No <AppendedData> block found")
        return

    payload = m.group(1)

    # VTK often uses '_' marker; remove whitespace but keep '_' if present
    payload_stripped = "".join(payload.split())

    print("payload starts with:", payload_stripped[:32])
    if payload_stripped.startswith("_"):
        payload_stripped = payload_stripped[1:]

    # base64 decode
    try:
        raw = base64.b64decode(payload_stripped, validate=True)
        print("base64 decode: OK, bytes:", len(raw))
        print("first 16 bytes:", raw[:16].hex())
    except Exception as e:
        print("base64 decode: FAIL:", repr(e))

if __name__ == "__main__":
    main()
