from typing import Optional
from data_models.handover_algorithm import HandoverAlgorithm
from data_models.latlng import LatLng
from data_models.q_network import QNetwork
from helpers.filters import Filters
from helpers.functions import Functions
from utils.location_utils import LocationUtils
from data_models.base_tower import BaseTower
from data_models.ng_ran_report import NGRANReport
from utils.wave_utils import WaveUtils
from colorama import Fore, init
from math import ceil
import torch
import numpy as np

init(autoreset=True)


class UserEquipment:

    min_time_of_stay: float = 1
    __model = None
    # Simplified RSRP proxy for Qout (TS 38.133 §8.1.1 / TS 36.133 §7.6).
    # Real Qout is SINR/BLER-based (10 % BLER of hypothetical PDCCH).
    # Normalized thresholds below map to ≈ −116 dBm for each RAT.
    rlf_threshold = {"NR": 41 / 127, "LTE": 25 / 97}
    # Handover interruption time (seconds) per RAT, intra-freq known cell.
    # NR: TS 38.133 §8.2.2 ≈ 20 ms   LTE: TS 36.133 §8.1.1.1 ≈ 40 ms
    __handover_time = {"NR": 0.02, "LTE": 0.04}

    def __init__(
        self,
        id: int,
        all_bs: list[BaseTower],
        latlng: Optional[LatLng] = None,  # can be null only if move_to() is used
        g_rx: float = 0.0,
        serving_bs: Optional[BaseTower] = None,
        print_logs_on_movement: bool = False,
        handover_algorithm: HandoverAlgorithm = HandoverAlgorithm.A3_RSRP_3GPP,
        cho_similarity_weight: float = 0.03,
        cho_q_weight: float = 0.97,
    ):
        self.id: int = id
        self.g_rx: float = g_rx  # 0 to +2 dBi
        self.latlng: LatLng = latlng
        self.serving_bs = serving_bs
        self.path_history: list[LatLng] = [] if latlng is None else [latlng]
        self.all_bs = all_bs
        self.print_logs_on_movement = print_logs_on_movement
        self.generated_reports: list[NGRANReport] = []
        # (bs_id, timestep) after each connection
        self.connection_history: list[tuple[int, float]] = []
        self.handover_algorithm = handover_algorithm
        self.cho_similarity_weight = cho_similarity_weight
        self.cho_q_weight = cho_q_weight
        self.speed = 0
        self.angle = 0
        self.rlf_count = 0
        self.dho_time = 0

        self.__in_rlf = False

    @classmethod
    def load_model(
        cls,
        model_path: str = "outputs/final_ddqn_model.pth",
        map_location: str = "cpu",
    ):
        cls.__model = QNetwork.from_state_dict(
            torch.load(
                model_path,
                map_location=map_location,
            )
        ).eval()

    def __repr__(self):
        return f"UserEquipment(id: {self.id}, latlng: {self.latlng}, serving_bs: {self.serving_bs.id if self.serving_bs else None})"

    def __str__(self):
        return f"UserEquipment(id: {self.id}, latlng: {self.latlng}, serving_bs: {self.serving_bs.id if self.serving_bs else None})"

    def toggle_report_print(self, value: bool):
        self.print_logs_on_movement = value

    def __append_path_history(self) -> None:
        self.path_history.append(self.latlng)

    def __append_generated_reports(self, report: NGRANReport):
        self.generated_reports.append(report)

    def get_total_handovers(self) -> int:
        """Returns the total number of handovers (excludes the initial connection)."""
        return max(0, len(self.connection_history) - 1)

    def get_total_pingpong(self) -> int:
        """Counts ping-pong handovers: A→B→A where time spent on B < min_time_of_stay (seconds)."""
        count = 0
        history = self.connection_history
        for i in range(len(history) - 2):
            bs_a, _ = history[i]
            bs_b, t_arrived_b = history[i + 1]
            bs_return, t_left_b = history[i + 2]
            if bs_a == bs_return and bs_a != bs_b:
                time_on_b = t_left_b - t_arrived_b
                if time_on_b < self.min_time_of_stay:
                    count += 1
        return count

    def get_pingpong_rate(self) -> float:
        """
        Calculates the Ping-Pong Handover Rate.
        Returns the ratio of ping-pong handovers to total handovers as a float (0.0 to 1.0).
        """
        total_handovers = self.get_total_handovers()

        # Prevent division by zero if the UE hasn't performed any handovers
        if total_handovers == 0:
            return 0.0

        total_pingpong = self.get_total_pingpong()

        return total_pingpong / total_handovers

    def get_time_since_last_handover(self, current_timestep: float) -> float:
        """Returns raw time in seconds since the last handover. Returns min_time_of_stay if no handover has occurred."""
        if len(self.connection_history) >= 2:
            _, last_ho_time = self.connection_history[-1]
            return current_timestep - last_ho_time
        return self.min_time_of_stay

    def get_normalized_time_since_last_handover(self, current_timestep: float) -> float:
        """Returns time since last handover normalized to [0, 1], clamped at min_time_of_stay. 0 = just switched, 1 = fully cooled down."""
        return min(
            self.get_time_since_last_handover(current_timestep) / self.min_time_of_stay,
            1.0,
        )

    def _check_rlf(self, report):
        if self.serving_bs:
            current_rsrp = WaveUtils.normalize_rsrp_index(
                report.rsrp_values.get(self.serving_bs.id, 0), self.serving_bs.radio
            )
            threshold = self.rlf_threshold.get(self.serving_bs.radio, 25 / 97)
            if current_rsrp < threshold:
                if not self.__in_rlf:
                    self.rlf_count += 1
                    self.__in_rlf = True
            else:
                self.__in_rlf = False

    def __on_movement(
        self,
        timestep: float,
        speed: float,
        angle: float,
    ) -> NGRANReport:
        self.__append_path_history()
        report = self.generate_report(all_bs=self.all_bs, timestep=timestep)
        self.__append_generated_reports(report)
        self._check_rlf(report)
        self.speed = speed
        self.angle = angle
        # Logs
        if self.print_logs_on_movement:
            print(report)

        # Check for handover
        match self.handover_algorithm:
            case HandoverAlgorithm.A3_RSRP_3GPP:
                target_bs = self.check_handover_3gpp_rsrp()
            case HandoverAlgorithm.DDQN_CHO:
                target_bs = self.check_handover_ddqn(
                    similarity_weight=self.cho_similarity_weight,
                    q_weight=self.cho_q_weight,
                )
            case HandoverAlgorithm.DDQN:
                target_bs = self.check_handover_ddqn_only()
            case HandoverAlgorithm.NONE:
                return report

        if target_bs:
            # Log handover decision, (or if the user connected for the first time)
            if self.serving_bs:
                if self.print_logs_on_movement:
                    print(
                        Fore.RED
                        + f"{self.id} handover from BS {self.serving_bs.id} to BS {target_bs.id} at {timestep}"
                    )
                self.handover(target_bs=target_bs, timestep=timestep)
            else:
                if self.print_logs_on_movement:
                    print(
                        Fore.MAGENTA
                        + f"UE {self.id} connecting to BS {target_bs.id} at {timestep}"
                    )
                self.connect_to_tower(bs=target_bs, timestep=timestep)

        return report

    def move_deg(
        self,
        lat_offset: float,
        long_offset: float,
        timestep: float,
        speed: float,
        angle: float,
    ):
        new_latitude = self.latlng.lat + lat_offset
        new_longitude = self.latlng.long + long_offset
        self.latlng = LatLng(lat=new_latitude, long=new_longitude)
        self.__on_movement(
            timestep=timestep,
            speed=speed,
            angle=angle,
        )

    def move_meters(
        self,
        distance: float,
        timestep: float,
        speed: float,
        angle: float = 0.0,
    ):
        new_point: LatLng = LocationUtils.move_meters(
            point=self.latlng, distance=distance, angle=angle
        )
        self.latlng = new_point
        self.__on_movement(timestep=timestep, speed=speed, angle=angle)

    def move_to(
        self, latlng: LatLng, timestep, speed: float, angle: float
    ) -> NGRANReport:
        self.latlng = latlng
        return self.__on_movement(
            timestep=timestep,
            speed=speed,
            angle=angle,
        )

    def generate_report(
        self,
        all_bs: list[BaseTower],
        timestep: float,
    ) -> NGRANReport:
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

    def check_handover_ddqn_only(self):
        """Pure DDQN: top-4 filtering → build state → argmax Q-value → return target tower."""
        if not self.generated_reports:
            return None
        report = self.generated_reports[-1]
        if not self.serving_bs:
            best_bs_id = max(report.rsrp_values, key=report.rsrp_values.get)
            return next((bs for bs in self.all_bs if bs.id == best_bs_id), None)
        # 1- Top-4 Filtering (with serving guarantee)
        top_4_towers = Filters.top_k_towers(all_bs=self.all_bs, report=report, k=4)
        if self.serving_bs not in top_4_towers:
            top_4_towers[-1] = self.serving_bs
        # 2- Build state (matches env obs space: rsrp, rsrp_trend, serving_one_hot, speed, time_since_ho)
        top_4_rsrp = [
            WaveUtils.normalize_rsrp_index(report.rsrp_values.get(bs.id, 0), bs.radio)
            for bs in top_4_towers
        ]
        if len(self.generated_reports) >= 2:
            prev_report = self.generated_reports[-2]
            rsrp_trend = [
                WaveUtils.normalize_rsrp_index(
                    report.rsrp_values.get(bs.id, 0), bs.radio
                )
                - WaveUtils.normalize_rsrp_index(
                    prev_report.rsrp_values.get(bs.id, 0), bs.radio
                )
                for bs in top_4_towers
            ]
        else:
            rsrp_trend = [0.0, 0.0, 0.0, 0.0]
        serving_one_hot = [0, 0, 0, 0]
        if self.serving_bs in top_4_towers:
            serving_one_hot[top_4_towers.index(self.serving_bs)] = 1
        norm_speed = min(self.speed / 30.0, 1.0)
        time_since_ho = self.get_normalized_time_since_last_handover(report.timestep)
        state = np.concatenate(
            [top_4_rsrp, rsrp_trend, serving_one_hot, [norm_speed], [time_since_ho]],
            dtype=np.float32,
        )
        # 3- DDQN argmax
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.__model(state_tensor)[0]
        action = q_vals.argmax().item()
        target_bs = top_4_towers[action]
        if target_bs == self.serving_bs:
            return None
        return target_bs

    def check_handover_ddqn(
        self,
        similarity_weight: float = 0.03,
        q_weight: float = 0.97,
    ):
        """DDQN + cosine similarity tiebreaker. State matches check_handover_ddqn_only."""
        weights = [similarity_weight, q_weight]
        if not self.generated_reports:
            return None
        report = self.generated_reports[-1]
        if not self.serving_bs:
            best_bs_id = max(report.rsrp_values, key=report.rsrp_values.get)
            return next((bs for bs in self.all_bs if bs.id == best_bs_id), None)
        # 1- Top-4 Filtering (with serving guarantee)
        top_4_towers = Filters.top_k_towers(all_bs=self.all_bs, report=report, k=4)
        if self.serving_bs not in top_4_towers:
            top_4_towers[-1] = self.serving_bs
        # 2- Build state (matches env obs space: rsrp, rsrp_trend, serving_one_hot, speed, time_since_ho)
        top_4_rsrp = [
            WaveUtils.normalize_rsrp_index(report.rsrp_values.get(bs.id, 0), bs.radio)
            for bs in top_4_towers
        ]
        if len(self.generated_reports) >= 2:
            prev_report = self.generated_reports[-2]
            rsrp_trend = [
                WaveUtils.normalize_rsrp_index(
                    report.rsrp_values.get(bs.id, 0), bs.radio
                )
                - WaveUtils.normalize_rsrp_index(
                    prev_report.rsrp_values.get(bs.id, 0), bs.radio
                )
                for bs in top_4_towers
            ]
        else:
            rsrp_trend = [0.0, 0.0, 0.0, 0.0]
        serving_one_hot = [0, 0, 0, 0]
        if self.serving_bs in top_4_towers:
            serving_one_hot[top_4_towers.index(self.serving_bs)] = 1
        norm_speed = min(self.speed / 30.0, 1.0)
        time_since_ho = self.get_normalized_time_since_last_handover(report.timestep)
        state = np.concatenate(
            [top_4_rsrp, rsrp_trend, serving_one_hot, [norm_speed], [time_since_ho]],
            dtype=np.float32,
        )
        # 3- DDQN raw Q-values
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = [q.item() for q in self.__model(state_tensor)[0]]
        # 4- Top 2 by raw Q-value
        indexed = list(enumerate(q_vals))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_2 = indexed[:2]

        tower1_idx, raw_q1 = top_2[0]
        tower2_idx, raw_q2 = top_2[1]
        softmax_qs = Functions.softmax_all([raw_q1, raw_q2])
        tower1_q = softmax_qs[0]
        tower2_q = softmax_qs[1]

        tower1 = top_4_towers[tower1_idx]
        tower2 = top_4_towers[tower2_idx]

        # 5- Cosine similarity tiebreaker
        angle_ue = self.angle
        similarity_tower1 = Functions.cos_similarity(
            angle_ue,
            Functions.bearing(pointA=self.latlng, pointB=tower1.latlng),
            normalized=True,
        )
        similarity_tower2 = Functions.cos_similarity(
            angle_ue,
            Functions.bearing(pointA=self.latlng, pointB=tower2.latlng),
            normalized=True,
        )

        score_tower_1 = Functions.weighted_sum([similarity_tower1, tower1_q], weights)
        score_tower_2 = Functions.weighted_sum([similarity_tower2, tower2_q], weights)

        # 6- Decision
        target_bs_idx = tower1_idx if score_tower_1 > score_tower_2 else tower2_idx
        target_bs: BaseTower = top_4_towers[target_bs_idx]

        if target_bs == self.serving_bs:
            return None
        return target_bs

    def check_handover_3gpp_rsrp(
        self,
        hysteresis: float = 2.0,
        time_to_trigger: float = 0.160,  # 160 ms
    ) -> Optional[BaseTower]:
        """Checks if a handover is needed based on 3GPP RSRP criteria.
        Comparison is done in dBm for fair inter-RAT evaluation."""
        # If serving bs is null, connect to the best available option
        last_report: NGRANReport = self.generated_reports[-1]
        if self.serving_bs is None:
            if not last_report.rsrp_values:
                return None
            best_bs_id = max(last_report.rsrp_values, key=last_report.rsrp_values.get)
            best_bs = next((bs for bs in self.all_bs if bs.id == best_bs_id), None)
            return best_bs

        bs_by_id = {bs.id: bs for bs in self.all_bs}

        # Compare in dBm so hysteresis (dB) applies fairly across LTE/NR
        serving_rsrp_dbm = WaveUtils.rsrp_index_to_dbm(
            last_report.rsrp_values[self.serving_bs.id], self.serving_bs.radio
        )

        best_bs_id = max(
            (bs_id for bs_id in last_report.rsrp_values if bs_id != self.serving_bs.id),
            key=lambda bs_id: WaveUtils.rsrp_index_to_dbm(
                last_report.rsrp_values[bs_id], bs_by_id[bs_id].radio
            ),
            default=None,
        )
        if best_bs_id is None:
            return None

        best_bs_radio = bs_by_id[best_bs_id].radio
        best_rsrp_dbm = WaveUtils.rsrp_index_to_dbm(
            last_report.rsrp_values[best_bs_id], best_bs_radio
        )

        # A3 entering condition: Mn > Ms + Hys (in dBm)
        if best_rsrp_dbm > serving_rsrp_dbm + hysteresis:
            # Check TTT: condition must hold for at least time_to_trigger seconds
            if len(self.generated_reports) > 1:
                delta_timestep = (
                    last_report.timestep - self.generated_reports[-2].timestep
                )
                if delta_timestep <= 0:  # Safety
                    return None
                # +1 because N reports span (N-1) intervals
                needed_reports = ceil(time_to_trigger / delta_timestep) + 1
                if len(self.generated_reports) >= needed_reports:
                    report_history = self.generated_reports[
                        -1 : -1 - needed_reports : -1
                    ]
                    is_satisfied = True
                    for report in report_history:
                        serving = WaveUtils.rsrp_index_to_dbm(
                            report.rsrp_values[self.serving_bs.id],
                            self.serving_bs.radio,
                        )
                        candidate = WaveUtils.rsrp_index_to_dbm(
                            report.rsrp_values[best_bs_id],
                            best_bs_radio,
                        )
                        if candidate <= serving + hysteresis:
                            is_satisfied = False
                            break
                    if is_satisfied:
                        return bs_by_id.get(best_bs_id)
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
        self.dho_time += self.__handover_time.get(target_bs.radio, 0.04)
