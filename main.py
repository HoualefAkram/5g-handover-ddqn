from user_equipment import UserEquipment
from base_tower import BaseTower
from latlng import LatLng
from wave_utils import WaveUtils


bs1 = BaseTower(
    id=0,
    latlng=LatLng(35.717583, -0.540996),  # ayachi
    connected_ues=[],
)

car = UserEquipment(
    id=0,
    latlng=LatLng(35.717122, -0.540052),  # Home
    serving_bs=bs1,  # starts connected to bs1
)

bs1.add_ue(ue=car)


rsrp1 = WaveUtils.calculate_rsrp(ue=car, bs=bs1)
print(f"rsrp1: {rsrp1:.2f} dBm")

car.move_meters(68, angle=270)


rsrp2 = WaveUtils.calculate_rsrp(ue=car, bs=bs1)
print(f"rsrp2: {rsrp2:.2f} dBm")


car.move_meters(56, angle=320)

rsrp3 = WaveUtils.calculate_rsrp(ue=car, bs=bs1)
print(f"rsrp3: {rsrp3:.2f} dBm")

car.move_meters(200, angle=270)


rsrp4 = WaveUtils.calculate_rsrp(ue=car, bs=bs1)
print(f"rsrp4: {rsrp4:.2f} dBm")
