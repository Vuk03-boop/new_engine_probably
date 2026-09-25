"""Estimate GPU memory for a list of DDS textures from their headers (no GPU upload).
usage: python dds_budget.py textures.txt
DDS stores the block-compressed mip chain as the GPU consumes it, so payload bytes
(file size minus headers) approximate resident size; driver alignment/padding is not included."""
import collections, os, struct, sys

DXGI = {71: "BC1", 72: "BC1_SRGB", 74: "BC2", 77: "BC3", 78: "BC3_SRGB", 80: "BC4", 83: "BC5", 95: "BC6H", 98: "BC7", 99: "BC7_SRGB", 28: "RGBA8", 29: "RGBA8_SRGB"}
paths = [l.strip().replace("\\", "/") for l in open(sys.argv[1]) if l.strip()]
total, missing, by_fmt, biggest, not_dds = 0, [], collections.Counter(), [], []
for p in paths:
    if not os.path.exists(p):
        missing.append(p); continue
    with open(p, "rb") as f:
        head = f.read(148)
    if head[:4] != b"DDS ":
        not_dds.append(p); continue
    h, w, mips = struct.unpack_from("<III", head, 12)[0], struct.unpack_from("<I", head, 16)[0], struct.unpack_from("<I", head, 28)[0]
    fourcc = head[84:88]
    hdr = 128
    if fourcc == b"DX10":
        fmt = DXGI.get(struct.unpack_from("<I", head, 128)[0], f"DXGI{struct.unpack_from('<I', head, 128)[0]}")
        hdr = 148
    else:
        fmt = fourcc.decode("latin1").strip("\0") or "uncompressed"
    size = os.path.getsize(p) - hdr
    total += size
    by_fmt[fmt] += size
    biggest.append((size, w, h, max(mips, 1), fmt, os.path.basename(p)))
mib = 1 / (1024 * 1024)
print(f"textures listed: {len(paths)}  found DDS: {len(biggest)}  missing: {len(missing)}  not DDS: {len(not_dds)}")
print(f"estimated resident size, all mips: {total * mib:.1f} MiB")
print("by format (MiB):", {k: round(v * mib, 1) for k, v in by_fmt.most_common()})
print("largest 5:", [(round(s * mib, 1), f"{w}x{h}", m, f, n) for s, w, h, m, f, n in sorted(biggest, reverse=True)[:5]])
for p in missing[:10]: print("missing:", p)
for p in not_dds[:10]: print("not DDS:", p)
