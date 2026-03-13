from typing import Optional
from data_models.latlng import LatLng
from utils.location_utils import LocationUtils
from typing import Optional
from data_models.base_tower import BaseTower
from data_models.ng_ran_report import NGRANReport
from utils.wave_utils import WaveUtils


class UserEquipment:

    def __init__(
        self,
        id: int,
        all_bs: list[BaseTower],
        latlng: Optional[LatLng] = None,  # can be null only if move_to() is used
        g_rx: float = 0.0,
        serving_bs: Optional[BaseTower] = None,
        print_report_on_movement: bool = False,
    ):
        self.id: int = id
        self.g_rx: float = g_rx  # 0 to +2 dBi
        self.latlng: LatLng = latlng
        self.serving_bs = serving_bs
        self.path_history: list[LatLng] = list(latlng) if (latlng) else list()
        self.all_bs = all_bs
        self.print_report_on_movement = print_report_on_movement

    def __repr__(self):
        return f"UserEquipment(id: {self.id}, latlng: {self.latlng}, serving_bs: {self.serving_bs.id})"

    def __str__(self):
        return f"UserEquipment(id: {self.id}, latlng: {self.latlng}, serving_bs: {self.serving_bs.id})"

    def toggle_report_print(self, value: bool):
        self.print_report_on_movement = value

    def __append_path_history(self) -> None:
        self.path_history.append(self.latlng)
        if self.print_report_on_movement:
            print(self.generate_report(all_bs=self.all_bs))

    def move_deg(self, lat_offset: float, long_offset: float):
        new_latitude = self.latlng.lat + lat_offset
        new_longitude = self.latlng.long + long_offset
        self.latlng = LatLng(lat=new_latitude, long=new_longitude)
        self.__append_path_history()

    def move_meters(self, distance: float, angle: float = 0.0):
        new_point: LatLng = LocationUtils.move_meters(
            point=self.latlng, distance=distance, angle=angle
        )
        self.latlng = new_point
        self.__append_path_history()

    def move_to(self, latlng: LatLng):
        self.latlng = latlng
        self.__append_path_history()

    def generate_report(self, all_bs: list[BaseTower]) -> NGRANReport:
        rsrp_values = {}
        rsrq_values = {}

        # calculate rsrp
        for bs in all_bs:
            rsrp_values[bs.id] = WaveUtils.calculate_rsrp(ue=self, bs=bs)

        # calculate rsrq
        all_rsrp_dBm = list(rsrp_values.values())

        for bs in all_bs:
            rsrq = WaveUtils.calculate_rsrq(
                serving_tower=bs,
                serving_rsrp=rsrp_values[bs.id],
                all_rsrp_dBm=all_rsrp_dBm,
            )
            rsrq_values[bs.id] = rsrq

        return NGRANReport(
            ue_id=self.id,
            rsrp_values=rsrp_values,
            rsrq_values=rsrq_values,
        )
