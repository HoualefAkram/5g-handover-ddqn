from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.path_gen import PathGeneration
from utils.map_downloader import MapDownloader
from utils.tower_downloader import TowerDownloader
from utils.render import Render
from utils.trace_parser import TraceParser


MAP_TOP_LEFT = LatLng(35.734904, -0.578253)
MAP_BOTTOM_RIGHT = LatLng(35.698884, -0.513860)
NUMBER_OF_UE = 2
# --- outputs ---
OSM_DOWNLOAD_PATH = "maps/map.osm"

MapDownloader.download_osm_by_bbox(
    top_left=MAP_TOP_LEFT,
    bottom_right=MAP_BOTTOM_RIGHT,
    output_file=OSM_DOWNLOAD_PATH,
)


bs_list: list[BaseTower] = TowerDownloader.get_towers_in_bbox(
    top_left=MAP_TOP_LEFT,
    bottom_right=MAP_BOTTOM_RIGHT,
)


cars = [
    UserEquipment(
        id=i,
        serving_bs=bs_list[0],  # starts connected to bs0
        all_bs=bs_list,
        print_report_on_movement=True,
    )
    for i in range(NUMBER_OF_UE)
]

for i in range(NUMBER_OF_UE):
    bs_list[0].add_ue(ue=cars[i])

path_gen = PathGeneration(stop_trip_generation_after=NUMBER_OF_UE)
path_gen.run()

vehicle_paths = TraceParser.parse_fcd_trace()

for id, path in vehicle_paths.items():
    car = cars[id]
    for point in path:
        car.move_to(point)

Render.render_map(bs_list=bs_list, ue_list=cars)
