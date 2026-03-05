from user_equipment import UserEquipment
from base_tower import BaseTower
from latlng import LatLng
from wave_utils import WaveUtils


bs1 = BaseTower(
    id=0,
    latlng=LatLng(36.7538, 3.0588),  # Algiers, Algeria
    connected_ues=[],
    p_tx=43.0,  # dBm — typical 5G macro BS transmit power
    frequency=3.5e9,  # Hz  — 3500 MHz, standard 5G sub-6GHz band
    g_tx=15.0,  # dBi — typical macro sector antenna gain
)

car = UserEquipment(
    id=0,
    latlng=LatLng(36.7520, 3.0560),  # ~300m away from BS1
    g_rx=0.0,  # dBi — omnidirectional phone/car antenna
    serving_bs=bs1,  # starts connected to bs1
)


rsrp1 = WaveUtils.calculate_rsrp(ue=car, bs=bs1)
print(f"rsrp1: {rsrp1:.2f} dBm")

car.move_meters(200, angle=90)

rsrp2 = WaveUtils.calculate_rsrp(ue=car, bs=bs1)
print(f"rsrp2: {rsrp2:.2f} dBm")
