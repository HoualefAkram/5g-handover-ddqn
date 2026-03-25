from typing import Optional
from data_models.handover_algorithm import HandoverAlgorithm
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
        handover_algorithm: HandoverAlgorithm = HandoverAlgorithm.A3_RSRP_3GPP,
    ):
        self.id: int = id
        self.g_rx: float = g_rx  # 0 to +2 dBi
        self.latlng: LatLng = latlng
        self.serving_bs = serving_bs
        self.path_history: list[LatLng] = [] if latlng is None else [latlng]
        self.all_bs = all_bs
        self.print_report_on_movement = print_report_on_movement
        self.generated_reports: list[NGRANReport] = []
        # (bs_id, timestep) after each connection
        self.connection_history: list[tuple[int, float]] = []
        self.handover_algorithm = handover_algorithm

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

    def get_total_handovers(self) -> int:
        """Returns the total number of handovers (excludes the initial connection)."""
        return max(0, len(self.connection_history) - 1)

    def get_total_pingpong(self, min_time_of_stay: float = 1.0) -> int:
        """Counts ping-pong handovers: A→B→A where time spent on B < min_time_of_stay (seconds)."""
        count = 0
        history = self.connection_history
        for i in range(len(history) - 2):
            bs_a, _ = history[i]
            bs_b, t_arrived_b = history[i + 1]
            bs_return, t_left_b = history[i + 2]
            if bs_a == bs_return and bs_a != bs_b:
                time_on_b = t_left_b - t_arrived_b
                if time_on_b < min_time_of_stay:
                    count += 1
        return count

    def get_pingpong_rate(self, min_time_of_stay: float = 1.0) -> float:
        """
        Calculates the Ping-Pong Handover Rate.
        Returns the ratio of ping-pong handovers to total handovers as a float (0.0 to 1.0).
        """
        total_handovers = self.get_total_handovers()

        # Prevent division by zero if the UE hasn't performed any handovers
        if total_handovers == 0:
            return 0.0

        total_pingpong = self.get_total_pingpong(min_time_of_stay=min_time_of_stay)

        return total_pingpong / total_handovers

    def __on_movement(self, timestep) -> NGRANReport:
        self.__append_path_history()
        report = self.generate_report(all_bs=self.all_bs, timestep=timestep)
        self.__append_generated_reports(report)
        # Logs
        if self.print_report_on_movement:
            print(report)

        # Check for handover
        match self.handover_algorithm:
            case HandoverAlgorithm.A3_RSRP_3GPP:
                target_bs = self.check_handover_3gpp_rsrp()
            case HandoverAlgorithm.DDQN_CHO:
                target_bs = self.check_handover_ddqn()

        if target_bs:
            # Log handover decision, (or if the user connected for the first time)
            if self.serving_bs:
                print(
                    Fore.RED
                    + f"{self.id} handover from BS {self.serving_bs.id} to BS {target_bs.id} at {timestep}"
                )
                self.handover(target_bs=target_bs, timestep=timestep)
            else:
                print(
                    Fore.MAGENTA
                    + f"UE {self.id} connecting to BS {target_bs.id} at {timestep}"
                )
                self.connect_to_tower(bs=target_bs, timestep=timestep)

        return report

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

    def move_to(self, latlng: LatLng, timestep) -> NGRANReport:
        self.latlng = latlng
        return self.__on_movement(timestep=timestep)

    def generate_report(self, all_bs: list[BaseTower], timestep: float) -> NGRANReport:
        rsrp_values = {}
        rsrq_values = {}

        # RSRP
        raw_rsrp_dbm = {}
        for bs in all_bs:
            raw_rsrp_dbm[bs.id] = WaveUtils.calculate_rsrp(ue=self, bs=bs)

        all_rsrp_list = list(raw_rsrp_dbm.values())

        # RSRQ
        for bs in all_bs:
            raw_rsrq_db = WaveUtils.calculate_rsrq(
                serving_tower=bs,
                serving_rsrp=raw_rsrp_dbm[bs.id],
                all_rsrp_dBm=all_rsrp_list,
            )

            radio_type = bs.radio

            # Map using the helper functions
            rsrp_values[bs.id] = WaveUtils.rsrp_to_index(
                raw_rsrp_dbm[bs.id], radio_type
            )
            rsrq_values[bs.id] = WaveUtils.rsrq_to_index(raw_rsrq_db, radio_type)

        return NGRANReport(
            ue_id=self.id,
            rsrp_values=rsrp_values,
            rsrq_values=rsrq_values,
            timestep=timestep,
        )

    def set_handover_algorithm(self, algorithm: HandoverAlgorithm):
        self.handover_algorithm = algorithm

    def check_handover_ddqn(self):
        # 1- Top-4 Filtering
        # 2- DDQN
        # 3- Softmax
        # 4- Top 2
        # 5- Weighted Sum
        # 6- Decision
        ...

    def check_handover_3gpp_rsrp(
        self,
        hysteresis: float = 3.0,
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
        if last_report.rsrp_values[best_bs_id] > serving_rsrp + hysteresis:
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
                        if candidate <= serving + hysteresis:
                            is_satisfied = False
                            break
                    if is_satisfied:
                        return next(
                            (bs for bs in self.all_bs if bs.id == best_bs_id), None
                        )
        return None

    def connect_to_tower(self, bs: BaseTower, timestep: float):
        """Called only if the user doesnt have a bse tower"""
        if self.serving_bs:
            return
        self.serving_bs = bs
        bs.add_ue(self)
        self.connection_history.append((bs.id, timestep))

    def handover(self, target_bs: BaseTower, timestep: float):
        """Performs handover to the target base station."""
        if self.serving_bs == target_bs:
            return
        if self.serving_bs:
            self.serving_bs.remove_ue(self.id)
        target_bs.add_ue(self)
        self.serving_bs = target_bs
        self.connection_history.append((target_bs.id, timestep))
