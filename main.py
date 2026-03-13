from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.path_gen import PathGeneration
from utils.map_downloader import MapDownloader
from utils.render import Render
from utils.trace_parser import TraceParser


MAP_TOP_LEFT = LatLng(35.734904, -0.578253)
MAP_BOTTOM_RIGHT = LatLng(35.698884, -0.513860)
NUMBER_OF_UE = 1
# --- outputs ---
OSM_DOWNLOAD_PATH = "maps/map.osm"

MapDownloader.download_osm_by_bbox(
    top_left=MAP_TOP_LEFT,
    bottom_right=MAP_BOTTOM_RIGHT,
    output_file=OSM_DOWNLOAD_PATH,
)


bs1 = BaseTower(
    id=1,
    latlng=LatLng(35.717583, -0.540996),  # top left
    connected_ues=[],
)


bs2 = BaseTower(
    id=2,
    latlng=LatLng(35.717558, -0.539236),  # top right
    connected_ues=[],
)

bs3 = BaseTower(
    id=3,
    latlng=LatLng(35.717580, -0.540077),  # center top
    connected_ues=[],
)


car = UserEquipment(
    id=0,
    serving_bs=bs2,  # starts connected to bs2
    all_bs=[bs1, bs2, bs3],
    print_report_on_movement=True,
)

bs2.add_ue(ue=car)

path_gen = PathGeneration(stop_trip_generation_after=1)
path_gen.run()

vehicle_paths = TraceParser.parse_fcd_trace()


car_paths = vehicle_paths[str(car.id)]

for path in car_paths:
    car.move_to(path)

Render.render_map(ue=car, output="output.html", bs_list=[bs1, bs2, bs3])
