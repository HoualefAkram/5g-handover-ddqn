from colorama import Fore, Style

from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.map_downloader import MapDownloader
from utils.path_gen import PathGeneration
from utils.tower_downloader import TowerDownloader

# --- Params ---
MAP_TOP_LEFT = LatLng(51.522004, -0.157535)  # London
MAP_BOTTOM_RIGHT = LatLng(51.512737, -0.134189)  # London
MCC = 234  # UK
OSM_DOWNLOAD_PATH = "cache/maps/map.osm"  # cache folder
SIMULATION_TIME = 600  # (10 minutes)
STEP_LENGTH = 0.15  # 15 ms
SEED = 200


# --- Execution ---

print(
    Fore.CYAN
    + Style.BRIGHT
    + f"--- Preparing Simulation for {SIMULATION_TIME} seconds ---"
)

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


path_gen = PathGeneration(
    end_simulation=SIMULATION_TIME,
    step_length=STEP_LENGTH,
    seed=SEED,
)
path_gen.run()


print(Fore.GREEN + Style.BRIGHT + f"--- Preparation Done! ---")
