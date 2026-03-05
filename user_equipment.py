from latlng import LatLng
from location_utils import LocationUtils
from typing import Optional
from base_tower import BaseTower


class UserEquipment:

    def __init__(
        self,
        id: int,
        latlng: LatLng,
        g_rx: float,
        serving_bs: Optional[BaseTower] = None,
    ):
        self.id: int = id
        self.g_rx: float = g_rx  # 0 to +2 dBi
        self.latlng: LatLng = latlng
        self.serving_bs = serving_bs

    def move_deg(self, lat_offset: float, long_offset: float):
        new_latitude = self.latlng.lat + lat_offset
        new_longitude = self.latlng.long + long_offset
        self.latlng = LatLng(lat=new_latitude, long=new_longitude)

    def move_meters(self, distance: float, angle: float = 0.0):
        new_point: LatLng = LocationUtils.move_meters(
            point=self.latlng, distance=distance, angle=angle
        )
        self.latlng = new_point

    def move_to(self, latlng: LatLng):
        self.latlng = latlng
