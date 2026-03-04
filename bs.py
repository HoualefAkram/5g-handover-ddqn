from ue import UE
from typing import Optional
from latlng import LatLng
from ng_ran_report import NGRANReport


class BS:

    def __init__(
        self,
        id: int,
        latlng: LatLng,
        connected_ues: list[UE],
        ng_ran_report: Optional[NGRANReport] = None,
    ):
        self.id = id
        self.latlng: LatLng = latlng
        self.connected_ues: list[UE] = connected_ues
        self.last_report: NGRANReport = ng_ran_report

    def receive_report(self, report: NGRANReport):
        self.last_report = report
