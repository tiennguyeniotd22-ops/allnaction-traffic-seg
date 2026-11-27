"""Pair source crop pixels with polygon overlays for instance QC."""
import argparse
from collections import defaultdict
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from overnight_drafts import COLORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--width", type=int, default=310)
    parser.add_argument("--height", type=int, default=175)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--number-objects", action="store_true")
    args = parser.parse_args()
    folder = args.folder
    manifest = json.loads((folder / "manifest.json").read_text())
    by_bucket = defaultdict(list)
    for row in manifest:
        if "roi" in row:
            by_bucket[row["selection"]].append(row)
    for bucket in by_bucket.values():
        bucket.sort(key=lambda r: (len(r["objects"]), -max(o["area640"] for o in r["objects"]), r["key"]))
    rows = [r for batch in zip_longest(*by_bucket.values()) for r in batch if r is not None]
    out = folder / "qc"
    out.mkdir(exist_ok=True)
    (out / "order.json").write_text(json.dumps([r["key"] for r in rows], indent=1))
    cols, height, width = args.cols, args.height, args.width
    per_sheet = cols * args.rows
    for first in range(0, len(rows), per_sheet):
        canvas = Image.new("RGB", (cols * width, args.rows * (height * 2 + 25)), "#ddd")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(rows[first:first + per_sheet]):
            x = j % cols * width
            y = j // cols * (height * 2 + 25)
            key = row["key"]
            with Image.open(folder / "images" / (key + ".jpg")) as original:
                raw = original.convert("RGB")
            raw.thumbnail((width - 2, height))
            canvas.paste(raw, (x, y))
            overlay = raw.copy()
            pen = ImageDraw.Draw(overlay)
            lines = (folder / "labels" / (key + ".txt")).read_text().splitlines()
            ids = []
            for number, line in enumerate(lines):
                fields = line.split()
                cls = int(fields[0])
                points = np.array(fields[1:], dtype=float).reshape(-1, 2) * raw.size
                points = [tuple(p) for p in points]
                pen.line(points + points[:1], fill=COLORS[cls], width=2)
                if args.number_objects and row["objects"][number]["area640"] >= 150:
                    center = np.mean(points, axis=0)
                    pen.text(tuple(center), str(number), fill="white", stroke_width=1, stroke_fill="black")
                ids.append(str(cls))
            canvas.paste(overlay, (x, y + height))
            draw.text((x + 2, y + 2 * height + 2), f"{first + j:03d} | {row['selection']} | {','.join(ids)}", fill="black")
        canvas.save(out / f"{first:03d}.jpg", quality=91)
    print(len(rows), out)


if __name__ == "__main__":
    main()
