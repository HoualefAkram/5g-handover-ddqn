from bs import BS
from ue import UE
from location_utils import LocationUtils
import numpy as np


class WaveUtils:
    __c = 299_792_458.0  # speed of light
    __n_los = 2.0  # straight line between BS and UE
    __n_nlos = 3.0  # buildings between BS and UE
    __los_threshold = 20  # 20 meters before there's a building blocking you

    @staticmethod
    def pd0(f_c: float, d0: float = 1):
        # PL(d0) = 20·log10(4π·d0·f_c / c)
        return 20 * np.log10(4 * np.pi * d0 * f_c / WaveUtils.__c)

    @staticmethod
    def path_loss(distance: float, n: float, f_c: float, d0: float = 1):
        return WaveUtils.pd0(d0=d0, f_c=f_c) + 10 * n * np.log10(distance / d0)

    @staticmethod
    def calculate_rsrp(bs: BS, ue: UE):
        # RSRP (dBm) = P_tx + G_tx + G_rx - PL(d) - L_shadow (TODO: add later) - L_fast (TODO: add later)
        # PL(d) = PL(d0) + 10·n·log10(d/d0)
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
