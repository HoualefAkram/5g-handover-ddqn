from colorama import Fore, Style

from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.map_downloader import MapDownloader
from utils.path_gen import PathGeneration
from utils.tower_downloader import TowerDownloader

# --- Params ---
# MAP_TOP_LEFT = LatLng(48.867010, 2.335946)  # FR, Paris (Train)
# MAP_BOTTOM_RIGHT = LatLng(48.847375, 2.352779)  # FR, Paris (Train)

MAP_TOP_LEFT = LatLng(51.519480, -0.169511)  # UK, London (Test)
MAP_BOTTOM_RIGHT = LatLng(51.479214, -0.105529)  # UK, London (Test)

MCC = 234  # UK
OSM_DOWNLOAD_PATH = "cache/maps/map.osm"  # cache folder
SIMULATION_TIME = 300  # (5 minutes)
STEP_LENGTH = 0.1  # 100 ms
SEED = 100
SPAWN_INTERVAL = 5


# --- Execution ---

if __name__ == "__main__":

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
        spawn_interval=SPAWN_INTERVAL,
    )
    path_gen.run()

    print(Fore.GREEN + Style.BRIGHT + f"--- Preparation Done! ---")
