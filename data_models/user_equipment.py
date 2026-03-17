from typing import Optional
from data_models.latlng import LatLng
from utils.location_utils import LocationUtils
from data_models.base_tower import BaseTower
from data_models.ng_ran_report import NGRANReport
from utils.wave_utils import WaveUtils
from colorama import Fore, init
from math import ceil

init(autoreset=True)


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
        self.path_history: list[LatLng] = [] if latlng is None else [latlng]
        self.all_bs = all_bs
        self.print_report_on_movement = print_report_on_movement
        self.generated_reports: list[NGRANReport] = []
        self.total_handovers: int = 0

    def __repr__(self):
        return f"UserEquipment(id: {self.id}, latlng: {self.latlng}, serving_bs: {self.serving_bs.id if self.serving_bs else None})"

    def __str__(self):
        return f"UserEquipment(id: {self.id}, latlng: {self.latlng}, serving_bs: {self.serving_bs.id if self.serving_bs else None})"

    def toggle_report_print(self, value: bool):
        self.print_report_on_movement = value

    def __append_path_history(self) -> None:
        self.path_history.append(self.latlng)

    def __append_generated_reports(self, report: NGRANReport):
        self.generated_reports.append(report)

    def __on_movement(self, timestep):
        self.__append_path_history()
        report = self.generate_report(all_bs=self.all_bs, timestep=timestep)
        self.__append_generated_reports(report)
        # Logs
        if self.print_report_on_movement:
            print(report)

        # Check for handover
        target_bs = self.check_handover_3gpp_rsrp()
        if target_bs:
            # Log handover decision, (or if the user connected for the first time)
            if self.serving_bs:
                print(
                    Fore.RED
                    + f"{self.id} handover from BS {self.serving_bs.id} to BS {target_bs.id}"
                )
                self.total_handovers += 1
            else:
                print(Fore.MAGENTA + f"UE {self.id} connecting to BS {target_bs.id}")
            self.handover(target_bs=target_bs)

    def move_deg(self, lat_offset: float, long_offset: float, timestep: float):
        new_latitude = self.latlng.lat + lat_offset
        new_longitude = self.latlng.long + long_offset
        self.latlng = LatLng(lat=new_latitude, long=new_longitude)
        self.__on_movement(timestep=timestep)

    def move_meters(self, distance: float, timestep: float, angle: float = 0.0):
        new_point: LatLng = LocationUtils.move_meters(
            point=self.latlng, distance=distance, angle=angle
        )
        self.latlng = new_point
        self.__on_movement(timestep=timestep)

    def move_to(self, latlng: LatLng, timestep):
        self.latlng = latlng
        self.__on_movement(timestep=timestep)

    def generate_report(self, all_bs: list[BaseTower], timestep: float) -> NGRANReport:
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
            timestep=timestep,
        )

    def check_handover_3gpp_rsrp(
        self,
        handover_threshold: float = 3.0,
        time_to_trigger: float = 3.0,
    ) -> Optional[BaseTower]:
        """Checks if a handover is needed based on 3GPP RSRP criteria."""
        # If serving bs is null, connect to the best available option
        last_report: NGRANReport = self.generated_reports[-1]
        if self.serving_bs is None:
            if not last_report.rsrp_values:
                return None
            best_bs_id = max(last_report.rsrp_values, key=last_report.rsrp_values.get)
            best_bs = next((bs for bs in self.all_bs if bs.id == best_bs_id), None)
            return best_bs
        serving_rsrp = last_report.rsrp_values[self.serving_bs.id]

        best_bs_id = max(
            (bs_id for bs_id in last_report.rsrp_values if bs_id != self.serving_bs.id),
            key=last_report.rsrp_values.get,
            default=None,
        )
        if best_bs_id is None:
            return None
        # Consider checking report history for TTT only if the HOM is satisfied
        if last_report.rsrp_values[best_bs_id] > serving_rsrp + handover_threshold:
            # check older reports if TTT is satisfied
            if len(self.generated_reports) > 1:
                # check how many reports are needed to satisfy the TTT
                delta_timestep = (
                    last_report.timestep - self.generated_reports[-2].timestep
                )
                if delta_timestep <= 0:  # Safety
                    return None
                needed_reports = ceil(time_to_trigger / delta_timestep)
                if len(self.generated_reports) >= needed_reports:
                    report_history = self.generated_reports[
                        -1 : -1 - needed_reports : -1
                    ]
                    is_satisfied = True
                    for report in report_history:
                        serving = report.rsrp_values[self.serving_bs.id]
                        candidate = report.rsrp_values[best_bs_id]
                        if candidate <= serving + handover_threshold:
                            is_satisfied = False
                            break
                    if is_satisfied:
                        return next(
                            (bs for bs in self.all_bs if bs.id == best_bs_id), None
                        )
        return None

    def handover(self, target_bs: BaseTower):
        """Performs handover to the target base station."""
        if self.serving_bs:
            self.serving_bs.remove_ue(self.id)
        target_bs.add_ue(self)
        self.serving_bs = target_bs
