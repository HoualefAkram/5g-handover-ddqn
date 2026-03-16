from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.path_gen import PathGeneration
from utils.map_downloader import MapDownloader
from utils.tower_downloader import TowerDownloader
from utils.render import Render
from utils.trace_parser import TraceParser


def run_simulation(
    top_left: LatLng,
    bottom_right: LatLng,
    num_ue: int,
    seed: int = 42,
    osm_download_path: str = "maps/map.osm",
) -> None:
    """Runs the complete city car simulator pipeline."""

    print(f"--- Starting Simulation for {num_ue} UEs ---")

    # 1. Download Map
    MapDownloader.download_osm_by_bbox(
        top_left=top_left,
        bottom_right=bottom_right,
        output_file=osm_download_path,
    )

    # 2. Fetch Base Stations
    bs_list: list[BaseTower] = TowerDownloader.get_towers_in_bbox(
        top_left=top_left,
        bottom_right=bottom_right,
    )

    if not bs_list:
        print("Error: No base stations found in this area. Exiting.")
        return

    # 3. Initialize User Equipment (Cars)
    cars: dict[int, UserEquipment] = {
        i: UserEquipment(
            id=i,
            all_bs=bs_list,
            print_report_on_movement=False,
        )
        for i in range(num_ue)
    }

    # 4. Generate Traffic Paths via SUMO
    path_gen = PathGeneration(stop_trip_generation_after=num_ue, seed=seed)
    path_gen.run()

    # 5. Parse Trace and Move Cars
    timesteps: list[dict[int, LatLng]] = TraceParser.parse_fcd_trace()

    print("--- Simulating Movement and Network Logic ---")
    for timestep in timesteps:
        for car_id, location in timestep.items():
            if car_id in cars:  # Safe check in case SUMO spawned extra vehicles
                car = cars[car_id]
                car.move_to(location)

    # 6. Render Final Map
    print("--- Rendering Final Output ---")
    Render.render_map(bs_list=bs_list, ue_list=list(cars.values()))

    for bs in bs_list:
        print(
            f"\033[35mBase Station {bs.id} served UEs: {[ue.id for ue in bs.connected_ues]}\033[0m"
        )


if __name__ == "__main__":
    # --- Configuration Parameters ---
    MAP_TOP_LEFT = LatLng(35.706161, -0.645196)
    MAP_BOTTOM_RIGHT = LatLng(35.697126, -0.630677)
    NUMBER_OF_UE = 1
    SEED = 100

    # Execute the pipeline
    run_simulation(
        top_left=MAP_TOP_LEFT,
        bottom_right=MAP_BOTTOM_RIGHT,
        num_ue=NUMBER_OF_UE,
        seed=SEED,
    )
