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
        p_tx: float = 43.0,
        frequency: float = 3.5e9,  # 4G LTE (common)2100 MHz, 4G LTE (low band)800 MHz, 5G sub-6GHz 3500 MHz, 5G mmWave 28 GHz
        bandwidth: float = 100e6,  # channel width, 20 MHz Minimum 5G deployment, 50 MHz Common 5G urban, 100 MHz Typical 5G urban, 200 MHz High-end 5G
        g_tx: float = 15,  # +14 to +17 dBi
        radio: Optional[str] = None,
    ):
        self.id = id
        self.latlng: LatLng = latlng
        self.connected_ues: list[UserEquipment] = connected_ues
        self.p_tx = p_tx
        self.g_tx = g_tx
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.radio = radio

    def __repr__(self):
        return f"BaseTower(id: {self.id}, connected_ues: {len(self.connected_ues)})"

    def __str__(self):
        return f"BaseTower(id: {self.id}, connected_ues: {len(self.connected_ues)})"

    def add_ue(self, ue: UserEquipment):
        self.connected_ues.append(ue)

    def remove_ue(self, ue_id: int):
        self.connected_ues = [ue for ue in self.connected_ues if ue.id != ue_id]
