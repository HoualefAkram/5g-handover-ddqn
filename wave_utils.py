from __future__ import annotations
from typing import TYPE_CHECKING
from base_tower import BaseTower
from location_utils import LocationUtils
import numpy as np

if TYPE_CHECKING:
    from user_equipment import UserEquipment


class WaveUtils:
    __c = 299_792_458.0  # speed of light
    __n_los = 2.0  # straight line between BS and UE
    __n_nlos = 3.0  # buildings between BS and UE
    __los_threshold = 20  # 20 meters before there's a building blocking you

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
    def calculate_rsrp(bs: BaseTower, ue: UserEquipment):
        # RSRP (dBm) = P_tx + G_tx + G_rx - PL(d) - L_shadow (TODO: add later) - L_fast (TODO: add later)
        # PL(d) = PL(d0) + 10 * n * log10(d/d0)
        # n: path loss exponent: NLOS 2.7 - 3.5, LOS 2 - 2.5
        p_tx = bs.p_tx  # transimition power
        g_tx = bs.g_tx  # transimition GAIN
        g_rx = ue.g_rx  # reception GAIN

        distance_ue_bs = LocationUtils.haversine(pointA=bs.latlng, pointB=ue.latlng)
        distance_ue_bs = max(distance_ue_bs, 1)  # protect against log(0)
        n = (
            WaveUtils.__n_los
            if distance_ue_bs <= WaveUtils.__los_threshold
            else WaveUtils.__n_nlos
        )

        pl = WaveUtils.path_loss(distance=distance_ue_bs, n=n, f_c=bs.frequency)

        rsrp = p_tx + g_tx + g_rx - pl
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
