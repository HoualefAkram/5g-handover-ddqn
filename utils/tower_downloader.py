import csv
import gzip
import json
import os
import requests
from pathlib import Path
from data_models.latlng import LatLng
from data_models.base_tower import BaseTower
from utils.location_utils import LocationUtils
from dotenv import load_dotenv
from colorama import Fore, Style, init

init(autoreset=True)

SUPPORTED_TOWERS = {"LTE", "NR"}
_TOWERS_CACHE_DIR = Path("cache/towers")
_TOWERS_JSON_CACHE = _TOWERS_CACHE_DIR / "towers.json"


class TowerDownloader:
    __DOWNLOAD_URL = "https://opencellid.org/ocid/downloads"

    @staticmethod
    def get_towers_in_bbox(
        top_left: LatLng, bottom_right: LatLng, mcc: int
    ) -> list[BaseTower]:
        min_lat = bottom_right.lat
        min_lon = top_left.long
        max_lat = top_left.lat
        max_lon = bottom_right.long

        if _TOWERS_JSON_CACHE.exists():
            with open(_TOWERS_JSON_CACHE, "r") as f:
                cached = json.load(f)
            if (
                LocationUtils.coords_are_identical(min_lat, cached["min_lat"])
                and LocationUtils.coords_are_identical(min_lon, cached["min_lon"])
                and LocationUtils.coords_are_identical(max_lat, cached["max_lat"])
                and LocationUtils.coords_are_identical(max_lon, cached["max_lon"])
            ):
                print(Fore.YELLOW + "Towers already fetched. Skipping...")
                return TowerDownloader.__parse_cells(cached["cells"])

        csv_path = TowerDownloader.__ensure_csv(mcc)
        cells = TowerDownloader.__filter_csv(
            csv_path, min_lat, min_lon, max_lat, max_lon
        )

        _TOWERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_TOWERS_JSON_CACHE, "w") as f:
            json.dump(
                {
                    "min_lat": min_lat,
                    "min_lon": min_lon,
                    "max_lat": max_lat,
                    "max_lon": max_lon,
                    "cells": cells,
                },
                f,
            )

        return TowerDownloader.__parse_cells(cells)

    @staticmethod
    def __ensure_csv(mcc: int = 234) -> Path:
        csv_path = _TOWERS_CACHE_DIR / f"cell_towers_{mcc}.csv.gz"

        if csv_path.exists():
            print(
                Fore.YELLOW
                + f"CSV database already downloaded at {csv_path}. Skipping..."
            )
            return csv_path

        load_dotenv()
        key = os.getenv("OPEN_CELL_ID_API_KEY")

        params = {"token": key, "type": "mcc", "file": f"{mcc}.csv.gz"}

        print(Fore.CYAN + f"--- Downloading OpenCellID CSV (MCC {mcc}) ---")

        response = requests.get(
            TowerDownloader.__DOWNLOAD_URL, params=params, stream=True
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "json" in content_type or "text" in content_type:
            body = response.text
            raise Exception(f"OpenCellID download failed: {body}")

        _TOWERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(csv_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r{Fore.CYAN}Downloading: {pct}%", end="", flush=True)

        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"\n{Fore.GREEN}Download complete: {csv_path} ({size_mb:.1f} MB)")
        return csv_path

    @staticmethod
    def __filter_csv(
        csv_path: Path,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> list[dict]:
        print(Fore.CYAN + "--- Filtering towers from local CSV database ---")
        cells = []

        fieldnames = [
            "radio",
            "mcc",
            "net",
            "area",
            "cell",
            "unit",
            "lon",
            "lat",
            "range",
            "samples",
            "changeable",
            "created",
            "updated",
            "averageSignal",
        ]

        with gzip.open(csv_path, "rt") as f:
            reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in reader:
                radio = row["radio"]
                if radio not in SUPPORTED_TOWERS:
                    continue
                lat = float(row["lat"])
                lon = float(row["lon"])
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    cells.append(
                        {
                            "cellid": int(row["cell"]),
                            "lat": lat,
                            "lon": lon,
                            "radio": radio,
                        }
                    )

        print(Fore.GREEN + f"Found {len(cells)} LTE/NR cells in bounding box")
        return cells

    @staticmethod
    def __parse_cells(cells: list[dict]) -> list[BaseTower]:
        towers: list[BaseTower] = []
        for cell in cells:
            tower = BaseTower(
                id=cell["cellid"] >> 8,
                latlng=LatLng(cell["lat"], cell["lon"]),
                radio=cell["radio"],
                connected_ues=[],
            )
            towers.append(tower)

        filtered = list(set(towers))
        print(
            Fore.GREEN
            + Style.BRIGHT
            + f"Got {len(towers)} LTE/NR Cells, {len(filtered)} BaseTowers"
        )
        return filtered
