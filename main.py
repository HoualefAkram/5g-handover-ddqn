from data_models.car_fcd_data import CarFcdData
from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.path_gen import PathGeneration
from utils.map_downloader import MapDownloader
from utils.tower_downloader import TowerDownloader
from utils.render import Render
from utils.fcd_parser import FcdParser
from colorama import Fore, Style, init
import webbrowser
from pathlib import Path
from utils.logger import Logger

init(autoreset=True)


def run_simulation(
    top_left: LatLng,
    bottom_right: LatLng,
    mcc: int,
    num_ue: int,
    seed: int = 42,
    osm_download_path: str = "cache/maps/map.osm",
    show_folium_output: bool = True,
    folium_output: str = "outputs/folium/simulation.html",
) -> None:

    logger = Logger()

    """Runs the complete city car simulator pipeline."""

    print(Fore.CYAN + Style.BRIGHT + f"--- Starting Simulation for {num_ue} UEs ---")

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
        mcc=mcc,
    )

    if not bs_list:
        print(
            Fore.RED
            + Style.BRIGHT
            + "Error: No base stations found in this area. Exiting."
        )
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
    fcd_data: list[dict[int, CarFcdData]] = FcdParser.parse_fcd_trace()

    print(Fore.CYAN + Style.BRIGHT + "--- Simulating Movement and Network Logic ---")
    for fcd in fcd_data:
        for car_id, car_data in fcd.items():
            if car_id in cars:  # Safe check in case SUMO spawned extra vehicles
                car = cars[car_id]
                report = car.move_to(car_data.latlng, timestep=car_data.timestep)
                rsrp = report.rsrp_values[car.serving_bs.id]
                rsrq = report.rsrq_values[car.serving_bs.id]
                logger.log_ue_metric(
                    ue_index=car.id,
                    metric=Logger.Metric.RSRP,
                    step=car_data.timestep,
                    value=rsrp,
                )
                logger.log_ue_metric(
                    ue_index=car.id,
                    metric=Logger.Metric.RSRQ,
                    step=car_data.timestep,
                    value=rsrq,
                )

    # 6. Render Final Map
    print(Fore.CYAN + Style.BRIGHT + "--- Rendering Final Output ---")
    Render.render_map(
        bs_list=bs_list, ue_list=list(cars.values()), output=folium_output
    )

    for bs in bs_list:
        print(
            Fore.BLUE
            + f"Base Station {bs.id} served UEs: {[ue.id for ue in bs.connected_ues]}"
        )
    print(
        Fore.RED
        + Style.BRIGHT
        + f"Total Handovers: {sum([ue.total_handovers for ue in cars.values()])}"
    )
    if show_folium_output:
        webbrowser.open(Path(folium_output).resolve())


if __name__ == "__main__":
    # --- Configuration Parameters ---
    MAP_TOP_LEFT = LatLng(51.465655, -1.018621)
    MAP_BOTTOM_RIGHT = LatLng(51.416325, -0.898452)
    MCC = 234
    NUMBER_OF_UE = 1
    SEED = 200

    # Execute the pipeline
    run_simulation(
        top_left=MAP_TOP_LEFT,
        bottom_right=MAP_BOTTOM_RIGHT,
        mcc=MCC,
        num_ue=NUMBER_OF_UE,
        seed=SEED,
    )
