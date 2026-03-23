from colorama import Fore, Style

from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.map_downloader import MapDownloader
from utils.path_gen import PathGeneration
from utils.tower_downloader import TowerDownloader

# --- Params ---
MAP_TOP_LEFT = LatLng(51.522004, -0.157535)
MAP_BOTTOM_RIGHT = LatLng(51.512737, -0.134189)
MCC = 234
OSM_DOWNLOAD_PATH = "cache/maps/map.osm"
MCC = 234
NUMBER_OF_UE = 1
SEED = 200


# --- Execution ---

print(Fore.CYAN + Style.BRIGHT + f"--- Preparing Simulation for {NUMBER_OF_UE} UEs ---")

MapDownloader.download_osm_by_bbox(
    top_left=MAP_TOP_LEFT,
    bottom_right=MAP_BOTTOM_RIGHT,
    output_file=OSM_DOWNLOAD_PATH,
)


bs_list: list[BaseTower] = TowerDownloader.download_towers_in_bbox(
    top_left=MAP_TOP_LEFT,
    bottom_right=MAP_BOTTOM_RIGHT,
    mcc=MCC,
)


path_gen = PathGeneration(stop_trip_generation_after=NUMBER_OF_UE, seed=SEED)
path_gen.run()


print(Fore.GREEN + Style.BRIGHT + f"--- Preparation Done! ---")
