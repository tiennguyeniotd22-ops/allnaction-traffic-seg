"""
AllnAction RoboTaxi Hackathon — Public Image Downloader for Missing Class 3 (Other vehicle)
Downloads CC-licensed / Public Domain photos of cyclos, tricycles, auto-rickshaws,
tuk-tuks, and trailers from Wikimedia Commons and writes a provenance manifest CSV.

Usage:
    python scripts/download_other_vehicle_images.py --out raw_other_vehicle --per-query 12
"""

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

QUERIES = [
    "cyclo hanoi vietnam street",
    "auto rickshaw traffic",
    "tuk tuk street",
    "cargo tricycle road",
    "tractor trailer truck highway",
]

HEADERS = {"User-Agent": "AllnActionHackathonBot/1.0 (https://github.com/tiennguyeniotd22-ops/allnaction-traffic-seg)"}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_wikimedia_images(out_dir: Path, per_query: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_titles = set()
    manifest_rows = []
    idx = 0

    for query in QUERIES:
        q_enc = urllib.parse.quote(query)
        search_url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&list=search"
            f"&srsearch={q_enc}&srnamespace=6&srlimit={per_query * 2}&format=json"
        )
        try:
            res = fetch_json(search_url)
        except Exception as e:
            print(f"[Warn] Search failed for '{query}': {e}")
            continue

        titles = [
            item["title"]
            for item in res.get("query", {}).get("search", [])
            if item["title"].lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        count_q = 0
        for title in titles:
            if title in seen_titles or count_q >= per_query:
                continue
            seen_titles.add(title)

            t_enc = urllib.parse.quote(title)
            info_url = (
                f"https://commons.wikimedia.org/w/api.php?action=query&titles={t_enc}"
                f"&prop=imageinfo&iiprop=url|extmetadata&iiurlwidth=1280&format=json"
            )
            try:
                info_res = fetch_json(info_url)
                pages = info_res.get("query", {}).get("pages", {})
                page = next(iter(pages.values()))
                iinfo = page.get("imageinfo", [{}])[0]
                dl_url = iinfo.get("thumburl") or iinfo.get("url")
                if not dl_url:
                    continue
                meta = iinfo.get("extmetadata", {})
                license_name = meta.get("LicenseShortName", {}).get("value", "Wikimedia Commons")

                ext = ".jpg" if dl_url.lower().endswith((".jpg", ".jpeg")) else ".png"
                fname = f"vn_other_{idx:03d}{ext}"
                req = urllib.request.Request(dl_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=20) as r, open(out_dir / fname, "wb") as f:
                    f.write(r.read())

                manifest_rows.append(
                    {
                        "filename": fname,
                        "query": query,
                        "title": title,
                        "source_url": iinfo.get("descriptionurl", dl_url),
                        "license": license_name,
                    }
                )
                idx += 1
                count_q += 1
            except Exception:
                continue

        print(f"Query '{query}': đã tải {count_q} ảnh")

    manifest_path = out_dir / "manifest_other_vehicle.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "query", "title", "source_url", "license"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTổng cộng đã tải {idx} ảnh vào {out_dir}/ (manifest: {manifest_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Other-Vehicle images from Wikimedia Commons")
    parser.add_argument("--out", type=Path, default=Path("raw_other_vehicle"))
    parser.add_argument("--per-query", type=int, default=10)
    args = parser.parse_args()
    download_wikimedia_images(args.out, args.per_query)


if __name__ == "__main__":
    main()
