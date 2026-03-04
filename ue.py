from latlng import LatLng
from utils import Utils
from typing import Optional
from bs import BS


class UE:

    def __init__(self, id: int, latlng: LatLng, serving_bs: Optional[BS] = None):
        self.id = id
        self.latlng: LatLng = latlng

    def move_deg(self, lat_offset: float, long_offset: float):
        new_latitude = self.latlng.lat + lat_offset
        new_longitude = self.latlng.long + long_offset
        self.latlng = LatLng(lat=new_latitude, long=new_longitude)

    def move_meters(self, distance: float, angle: float):
        new_point: LatLng = Utils.move_meters(
            point=self.latlng, distance=distance, angle=angle
        )
        self.latlng = new_point
