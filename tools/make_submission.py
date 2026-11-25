"""Packs saved prediction masks for an evaluation server (standard library only, run on the host). Input: the masks folder of one setting
(`<out>/masks/<setting>/<video>/<frame>.png`, written with VOS_SAVE_MASKS=1 on the LVOS V2 test split: every frame, palette of the first-frame
annotation). Output: a .zip with `Annotations/<video>/<frame>.png` (the layout the LVOS CodaLab scorer reads: res/Annotations/<video>; unconfirmed on the Evaluation tab). Check
the server's Evaluation tab for its exact format first; this tool only repacks, it never changes a mask.
Usage: python3 tools/make_submission.py <masks setting dir> <expected videos dir (test/JPEGImages)> <out.zip>"""
import sys
import struct
import zlib
import zipfile
from pathlib import Path


def _chunks(data):
    i = 8
    while i < len(data):
        n, t = struct.unpack(">I4s", data[i:i + 8])
        yield t, data[i + 8:i + 8 + n]
        i += 12 + n


def _blank_like(path):
    """An all-background PNG with the size, bit depth and palette of the mask at `path` (standard library only)."""
    data = Path(path).read_bytes()
    ch = dict(_chunks(data))
    w, h, depth, ctype = struct.unpack(">IIBB", ch[b"IHDR"][:10])
    assert ctype == 3 and depth == 8, "expected an 8-bit indexed PNG"
    raw = zlib.compress((b"\x00" + bytes(w)) * h, 9)
    out = b"\x89PNG\r\n\x1a\n"
    for t, d in ((b"IHDR", ch[b"IHDR"]), (b"PLTE", ch.get(b"PLTE", b"")), (b"IDAT", raw), (b"IEND", b"")):
        if t == b"PLTE" and not d:
            continue
        out += struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return out


def main(masks, jpegs, out):
    masks, jpegs = Path(masks), Path(jpegs)
    videos = sorted(p.name for p in jpegs.iterdir() if p.is_dir())
    missing = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for v in videos:
            frames = sorted(p.stem for p in (jpegs / v).glob("*.jpg"))
            have = {p.stem for p in (masks / v).glob("*.png")} if (masks / v).is_dir() else set()
            lost = [f for f in frames if f not in have]
            if lost:
                missing.append((v, len(lost), len(frames)))   # filled with background below
            ref = next(iter(sorted(have)), None)
            for f in frames:
                if f in have:
                    z.write(masks / v / f"{f}.png", f"Annotations/{v}/{f}.png")
                elif ref is not None:                      # frames before the first appearance are not predicted: all background
                    z.writestr(f"Annotations/{v}/{f}.png", _blank_like(masks / v / f"{ref}.png"))
    print(f"{out}: {len(videos)} videos; videos with missing frames: {len(missing)}")
    for m in missing[:10]:
        print("  missing", m)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
