from __future__ import annotations
from typing import TYPE_CHECKING
from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.location_utils import LocationUtils
import numpy as np
import math

if TYPE_CHECKING:
    from data_models.user_equipment import UserEquipment


class WaveUtils:
    __c = 299_792_458.0  # speed of light
    __n_los = 2.0  # straight line between BS and UE
    __n_nlos = 3.5  # buildings between BS and UE, dense urban (Rappaport range 2.7–3.5)
    __los_threshold = 5  # 5 meters before there's an object blocking you

    # Shadow fading (log-normal): std dev in dB per environment
    __shadow_std_los = 4.0  # dB, typical urban LOS
    __shadow_std_nlos = 8.03  # dB, 3GPP TR 38.901 Table 7.5-6 UMa-NLOS
    __shadow_decorrelation_dist = 37.0  # meters, 3GPP TR 38.901 Table 7.6.3.1-2 UMa

    # Shadow fading state: keyed by (ue_id, bs_id) -> (last_position, last_shadow_value)
    __shadow_state: dict[tuple[int, int], tuple[LatLng, float]] = {}

    # Fast fading RNG: keyed by (ue_id, bs_id) for reproducibility per link
    __fast_fading_rng: dict[tuple[int, int], np.random.RandomState] = {}
    __rng_seed_base: int = 200

    @staticmethod
    def reset_fading_state():
        """Clears all fading state. Call between simulation runs."""
        WaveUtils.__shadow_state.clear()
        WaveUtils.__fast_fading_rng.clear()

    @staticmethod
    def get_resource_blocks(bandwidth_hz: float) -> int:
        rb_table = {
            5e6: 25,
            10e6: 52,
            15e6: 79,
            20e6: 100,
            25e6: 128,
            50e6: 270,
            100e6: 132,  # 5G NR sub-6GHz
            200e6: 264,
        }
        return rb_table.get(bandwidth_hz, 100)  # default 100

    @staticmethod
    def pd0(f_c: float, d0: float = 1):
        # PL(d0) = 20 * log10(4 pi * d0 * f_c / c)
        return 20 * np.log10(4 * np.pi * d0 * f_c / WaveUtils.__c)

    @staticmethod
    def path_loss(distance: float, n: float, f_c: float, d0: float = 1):
        return WaveUtils.pd0(d0=d0, f_c=f_c) + 10 * n * np.log10(distance / d0)

    @staticmethod
    def _get_link_rng(ue_id: int, bs_id: int) -> np.random.RandomState:
        """Returns a per-link RNG seeded deterministically from (ue_id, bs_id)."""
        key = (ue_id, bs_id)
        if key not in WaveUtils.__fast_fading_rng:
            seed = WaveUtils.__rng_seed_base + ue_id * 10_000 + bs_id
            WaveUtils.__fast_fading_rng[key] = np.random.RandomState(seed)
        return WaveUtils.__fast_fading_rng[key]

    @staticmethod
    def calculate_shadow_fading(
        ue_id: int, bs_id: int, ue_pos: LatLng, is_los: bool
    ) -> float:
        """
        Correlated log-normal shadow fading (Gudmundson model).
        Returns shadow fading loss in dB.
        The value is spatially correlated: it evolves smoothly as the UE moves,
        with new independent samples blended in based on distance traveled.
        """
        key = (ue_id, bs_id)
        sigma = WaveUtils.__shadow_std_los if is_los else WaveUtils.__shadow_std_nlos
        rng = WaveUtils._get_link_rng(ue_id, bs_id)

        if key not in WaveUtils.__shadow_state:
            # First call for this link: draw an initial sample
            initial_value = rng.normal(0, sigma)
            WaveUtils.__shadow_state[key] = (ue_pos, initial_value)
            return initial_value

        last_pos, last_value = WaveUtils.__shadow_state[key]
        distance_moved = LocationUtils.haversine(pointA=last_pos, pointB=ue_pos)

        # Gudmundson autocorrelation: r = exp(-d / d_corr)
        d_corr = WaveUtils.__shadow_decorrelation_dist
        r = math.exp(-distance_moved / d_corr)

        # Correlated update: S_new = r * S_old + sqrt(1 - r^2) * N(0, sigma)
        new_value = r * last_value + math.sqrt(1 - r * r) * rng.normal(0, sigma)
        WaveUtils.__shadow_state[key] = (ue_pos, new_value)
        return new_value

    @staticmethod
    def calculate_fast_fading(ue_id: int, bs_id: int, is_los: bool) -> float:
        """
        Fast fading (small-scale) in dB.
        - LOS: Rician fading (K=9 dB), 3GPP TR 38.901 Table 7.5-6 UMi-LOS
        - NLOS: Rayleigh fading (no dominant path)
        Returns fading value in dB (can be positive or negative).
        """
        rng = WaveUtils._get_link_rng(ue_id, bs_id)

        if is_los:
            # Rician fading with K-factor = 7 dB (linear ~5.01), TR 38.901 Table 7.5-6 UMa-LOS
            k_db = 7.0
            k_lin = 10 ** (k_db / 10)
            # Rician envelope: dominant component + scattered
            # nu = sqrt(K / (K+1)), sigma = sqrt(1 / (2*(K+1)))
            nu = math.sqrt(k_lin / (k_lin + 1))
            s = math.sqrt(1 / (2 * (k_lin + 1)))
            x = rng.normal(nu, s)
            y = rng.normal(0, s)
            envelope = math.sqrt(x * x + y * y)
        else:
            # Rayleigh fading: envelope = sqrt(X^2 + Y^2), X,Y ~ N(0, 1/sqrt(2))
            s = 1.0 / math.sqrt(2)
            x = rng.normal(0, s)
            y = rng.normal(0, s)
            envelope = math.sqrt(x * x + y * y)

        # Convert envelope to dB (relative to mean power = 1)
        envelope = max(envelope, 1e-10)  # protect log
        return 20 * math.log10(envelope)

    @staticmethod
    def calculate_rsrp(bs: BaseTower, ue: UserEquipment):
        # RSRP (dBm) = P_tx + G_tx + G_rx - PL(d) - L_shadow + L_fast
        # PL(d) = PL(d0) + 10 * n * log10(d/d0)
        # n: path loss exponent: NLOS 2.7 - 3.5, LOS 2 - 2.5
        p_tx = bs.p_tx  # transmission power
        g_tx = bs.g_tx  # transmission GAIN
        g_rx = ue.g_rx  # reception GAIN

        distance_ue_bs = LocationUtils.haversine(pointA=bs.latlng, pointB=ue.latlng)
        distance_ue_bs = max(distance_ue_bs, 1)  # protect against log(0)
        is_los = distance_ue_bs <= WaveUtils.__los_threshold
        n = WaveUtils.__n_los if is_los else WaveUtils.__n_nlos

        pl = WaveUtils.path_loss(distance=distance_ue_bs, n=n, f_c=bs.frequency)
        l_shadow = WaveUtils.calculate_shadow_fading(ue.id, bs.id, ue.latlng, is_los)
        l_fast = WaveUtils.calculate_fast_fading(ue.id, bs.id, is_los)

        rsrp = p_tx + g_tx + g_rx - pl - l_shadow + l_fast
        return rsrp

    @staticmethod
    def calculate_rssi(
        all_rsrp_dBm: list[float],
        bandwidth_hz: float,
        noise_figure_db: float = 7.0,  # In a perfect world, the UE antenna would receive signals with zero added noise. In reality, the UE's hardware (amplifiers, circuits) adds some noise on top of what it receives. `noise_figure_db` measures how much extra noise the hardware adds.
    ) -> float:
        # RSSI (dBm) = BS signals (dominant) + thermal noise + UE interference (negligible)
        # convert to linear then sum: total = 10^(RSRP_bs1/10) + 10^(RSRP_bs2/10) + 10^(noise/10)
        # convert back to dBm: RSSI  = 10 * log10(total)
        # Thermal noise floor
        noise_dBm = -174 + 10 * np.log10(bandwidth_hz) + noise_figure_db
        # Convert all signals + noise to linear and sum
        total_linear = sum(10 ** (rsrp / 10) for rsrp in all_rsrp_dBm)
        total_linear += 10 ** (noise_dBm / 10)

        return 10 * np.log10(total_linear)

    @staticmethod
    def calculate_rsrq(
        serving_tower: BaseTower,
        serving_rsrp: float,
        all_rsrp_dBm: list[float],
    ):
        # 10 * np.log10(n) + serving_rsrp - rssi (*n is removed, since rsrp returns the total Power, not only per block)
        rssi = WaveUtils.calculate_rssi(
            all_rsrp_dBm=all_rsrp_dBm,
            bandwidth_hz=serving_tower.bandwidth,
        )

        return serving_rsrp - rssi

    @staticmethod
    def rsrp_to_index(rsrp_dbm: float, radio_type) -> int:
        """Converts raw RSRP (dBm) to 3GPP index based on radio technology."""
        radio_type = radio_type.upper()
        if radio_type == "LTE":
            # LTE RSRP Mapping (TS 36.133): Range 0 to 97
            return max(0, min(97, math.floor(rsrp_dbm + 141)))
        elif radio_type == "NR":
            # NR RSRP Mapping (TS 38.133): Range 0 to 127
            return max(0, min(127, math.floor(rsrp_dbm + 157)))

    @staticmethod
    def rsrq_to_index(rsrq_db: float, radio_type: str = "NR") -> int:
        """Converts raw RSRQ (dB) to 3GPP index based on radio technology."""
        radio_type = radio_type.upper()
        if radio_type == "LTE":
            # LTE RSRQ Mapping (TS 36.133): Range 0 to 34
            return max(0, min(34, math.floor((rsrq_db + 20.0) / 0.5)))
        elif radio_type == "NR":
            # NR RSRQ Mapping (TS 38.133): Range 0 to 127
            return max(0, min(127, math.floor((rsrq_db + 43.5) / 0.5)))

    @staticmethod
    def normalize_rsrp_index(rsrp_index: int, radio_type):
        return rsrp_index / 127 if radio_type == "NR" else rsrp_index / 97

    @staticmethod
    def normalize_rsrq_index(rsrq_index: int, radio_type):
        return rsrq_index / 127 if radio_type == "NR" else rsrq_index / 34
