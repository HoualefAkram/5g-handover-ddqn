from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from data_models.latlng import LatLng
from data_models.ng_ran_report import NGRANReport

if TYPE_CHECKING:
    from data_models.user_equipment import UserEquipment


class BaseTower:

    def __init__(
        self,
        id: int,
        latlng: LatLng,
        connected_ues: list[UserEquipment],
        p_tx: float,
        frequency: float,
        bandwidth: float,
        g_tx: float,
        radio: str,
    ):
        self.id = id
        self.latlng: LatLng = latlng
        self.connected_ues: list[UserEquipment] = connected_ues
        self.p_tx = p_tx
        self.g_tx = g_tx
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.radio = radio

    @classmethod
    def LTE(cls, id: int, latlng: LatLng, connected_ues: list[UserEquipment] = None):
        """
        UK LTE macro cell defaults (Band 3 — 1800 MHz).
        - P_tx: 46 dBm (40W), 3GPP TS 36.104 Table 6.2.1-1 (Wide Area BS)
        - G_tx: 15 dBi, typical 3-sector panel antenna
        - Frequency: 1800 MHz (Band 3, most deployed UK LTE band)
        - Bandwidth: 20 MHz (max LTE carrier, standard UK deployment)
        """
        return cls(
            id=id,
            latlng=latlng,
            connected_ues=connected_ues or [],
            p_tx=46.0,
            frequency=1800e6,
            bandwidth=20e6,
            g_tx=15.0,
            radio="LTE",
        )

    @classmethod
    def NR(cls, id: int, latlng: LatLng, connected_ues: list[UserEquipment] = None):
        """
        UK 5G NR macro cell defaults (n78 — 3500 MHz sub-6 GHz).
        - P_tx: 43 dBm (20W per carrier), 3GPP TS 38.104 Table 6.2.1-1 (Wide Area BS)
        - G_tx: 17 dBi, massive MIMO panel (typical 64T64R beamforming gain)
        - Frequency: 3500 MHz (n78, primary UK 5G band — EE, Three, Vodafone)
        - Bandwidth: 100 MHz (typical n78 allocation per operator)
        """
        return cls(
            id=id,
            latlng=latlng,
            connected_ues=connected_ues or [],
            p_tx=43.0,
            frequency=3500e6,
            bandwidth=100e6,
            g_tx=17.0,
            radio="NR",
        )

    def __repr__(self):
        return f"BaseTower(id: {self.id}, connected_ues: {len(self.connected_ues)})"

    def __str__(self):
        return f"BaseTower(id: {self.id}, connected_ues: {len(self.connected_ues)})"

    def add_ue(self, ue: UserEquipment):
        self.connected_ues.append(ue)

    def remove_ue(self, ue_id: int):
        self.connected_ues = [ue for ue in self.connected_ues if ue.id != ue_id]

    def __eq__(self, other):
        if not isinstance(other, BaseTower):
            return False

        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
