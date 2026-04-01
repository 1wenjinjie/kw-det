from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import requests


STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
ASSET_KEYS = [
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "nir09",
    "swir16",
    "swir22",
    "visual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Finland 2025 Sentinel-2 L2A imagery matching the local GPKG annotations."
    )
    parser.add_argument(
        "--annotations-dir",
        default="data/finland_2025_annotations",
        help="Directory containing Finland GPKG annotation files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/finland_2025_imagery/l2a",
        help="Directory to store downloaded L2A imagery.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent download workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def discover_products(annotations_dir: Path) -> List[Dict[str, str]]:
    products: List[Dict[str, str]] = []
    for gpkg in sorted(annotations_dir.glob("*.gpkg")):
        tile = gpkg.stem
        layers = gpd.list_layers(gpkg)["name"].tolist()
        for layer in layers:
            date = f"{layer[:4]}-{layer[4:6]}-{layer[6:8]}"
            products.append(
                {
                    "tile": tile,
                    "layer": layer,
                    "date": date,
                    "grid_code": f"MGRS-{tile}",
                }
            )
    return products


def search_item(session: requests.Session, grid_code: str, date: str, timeout: int) -> Dict:
    payload = {
        "collections": ["sentinel-2-l2a"],
        "query": {"grid:code": {"eq": grid_code}},
        "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
        "limit": 2,
    }
    response = session.post(STAC_SEARCH_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise RuntimeError(f"No L2A item found for {grid_code} on {date}")
    return features[0]


def expected_size(session: requests.Session, url: str, timeout: int) -> int:
    response = session.head(url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    return int(response.headers.get("Content-Length", "0"))


def download_one(session: requests.Session, url: str, dest: Path, timeout: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def ensure_asset(session: requests.Session, asset: Dict, product_dir: Path, timeout: int) -> Dict[str, str | int]:
    key = asset["key"]
    url = asset["href"]
    filename = asset["filename"]
    dest = product_dir / filename
    size = expected_size(session, url, timeout)
    if dest.exists() and dest.stat().st_size == size:
        return {"key": key, "path": str(dest), "bytes": size, "status": "exists"}

    product_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {dest}", flush=True)
    download_one(session, url, dest, timeout)
    return {"key": key, "path": str(dest), "bytes": size, "status": "downloaded"}


def main() -> int:
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_dir.exists():
        print(f"Annotation directory not found: {annotations_dir}", file=sys.stderr)
        return 1

    products = discover_products(annotations_dir)
    manifest: List[Dict] = []
    session = requests.Session()

    print(f"[info] discovered {len(products)} Finland acquisitions", flush=True)

    for product in products:
        item = search_item(session, product["grid_code"], product["date"], args.timeout)
        item_id = item["id"]
        product_dir = output_dir / item_id
        assets = []
        for key in ASSET_KEYS:
            asset = item["assets"][key]
            href = asset["href"]
            suffix = Path(href).suffix or ".tif"
            if key == "visual" and suffix == ".tif":
                filename = "TCI.tif"
            else:
                filename = Path(href).name
            assets.append({"key": key, "href": href, "filename": filename})

        item_path = product_dir / "item.json"
        product_dir.mkdir(parents=True, exist_ok=True)
        item_path.write_text(json.dumps(item, indent=2), encoding="utf-8")

        manifest.append(
            {
                "tile": product["tile"],
                "layer": product["layer"],
                "date": product["date"],
                "grid_code": product["grid_code"],
                "item_id": item_id,
                "product_uri": item["properties"].get("s2:product_uri"),
                "product_dir": str(product_dir),
                "assets": assets,
            }
        )

    manifest_path = output_dir / "finland_l2a_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[info] wrote manifest to {manifest_path}", flush=True)

    tasks = []
    for entry in manifest:
        product_dir = Path(entry["product_dir"])
        for asset in entry["assets"]:
            tasks.append((product_dir, asset))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for product_dir, asset in tasks:
            future = executor.submit(
                ensure_asset,
                requests.Session(),
                asset,
                product_dir,
                args.timeout,
            )
            future_map[future] = (product_dir, asset["key"])

        for future in as_completed(future_map):
            product_dir, key = future_map[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[done] {product_dir.name} {key} {result['status']} {result['bytes']} bytes",
                    flush=True,
                )
            except Exception as exc:
                print(f"[error] {product_dir.name} {key}: {exc}", file=sys.stderr, flush=True)
                return 1

    summary = {
        "products": len(manifest),
        "assets": len(results),
        "total_bytes": sum(int(r["bytes"]) for r in results),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
