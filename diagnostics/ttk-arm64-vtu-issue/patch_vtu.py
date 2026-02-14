from pathlib import Path

# Input file
input_file = Path("ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu")
output_file = Path("ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd_PATCHED.vtu")

print(f"Reading: {input_file}")

# Read as bytes
b = input_file.read_bytes()

# Find the AppendedData tag
tag = b'<AppendedData encoding="base64">'
i = b.find(tag)
assert i != -1, "Could not find AppendedData tag"

k = i + len(tag)  # Position right after the closing '>'

# Find first underscore after the tag
j = b.find(b'_', k)
assert j != -1, "Could not find underscore marker"

# Show what we're removing
removed = b[k:j]
print(f"Removing {len(removed)} bytes between '>' and '_': {repr(removed)}")

# Patch: remove whitespace between '>' and '_'
patched = b[:k] + b'_' + b[j+1:]

# Write patched file
output_file.write_bytes(patched)
print(f"✓ Wrote patched file: {output_file}")
print(f"  Original size: {len(b)} bytes")
print(f"  Patched size: {len(patched)} bytes")
print(f"  Difference: {len(b) - len(patched)} bytes removed")
