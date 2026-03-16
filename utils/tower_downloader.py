import requests
from data_models.latlng import LatLng
from data_models.base_tower import BaseTower
import os
from dotenv import load_dotenv
from colorama import Fore, Style, init

init(autoreset=True)

SUPPORTED_TOWERS = {"LTE", "NR"}


class TowerDownloader:
    __OPEN_CELL_ID_BASE_URL = "https://opencellid.org"

    @staticmethod
    def get_towers_in_bbox(top_left: LatLng, bottom_right: LatLng) -> list[BaseTower]:
        min_lat = bottom_right.lat
        min_lon = top_left.long
        max_lat = top_left.lat
        max_lon = bottom_right.long

        load_dotenv()
        OPENCELLID_KEY = os.getenv("OPEN_CELL_ID_API_KEY")

        print(Fore.CYAN + "--- Fetching Base Towers from OpenCellID ---")

        response = requests.get(
            f"{TowerDownloader.__OPEN_CELL_ID_BASE_URL}/cell/getInArea",
            params={
                "key": OPENCELLID_KEY,
                "BBOX": f"{min_lat},{min_lon},{max_lat},{max_lon}",
                "format": "json",
            },
        )

        response.raise_for_status()
        data = response.json()
        if data.get("error", False):
            error_text = f"Failed to get Base Towers. status_code: {response.status_code}, error: {data.get("error", None)},code: {data.get("code", None)}"
            print(Fore.RED + error_text)
            raise Exception(error_text)

        towers: list[BaseTower] = []
        for cell in data.get("cells", []):
            radio = cell["radio"]
            if radio not in SUPPORTED_TOWERS:
                continue
            tower = BaseTower(
                id=cell["cellid"],
                latlng=LatLng(cell["lat"], cell["lon"]),
                radio=cell["radio"],
                connected_ues=[],
            )
            towers.append(tower)

        print(Fore.GREEN + Style.BRIGHT + f"Fetched {len(towers)} LTE/NR BaseTowers")

        return towers
