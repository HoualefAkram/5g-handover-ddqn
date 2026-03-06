from user_equipment import UserEquipment
from base_tower import BaseTower
from latlng import LatLng
from wave_utils import WaveUtils


bs1 = BaseTower(
    id=1,
    latlng=LatLng(35.717583, -0.540996),  # top left
    connected_ues=[],
)


bs2 = BaseTower(
    id=2,
    latlng=LatLng(35.717558, -0.539236),  # top right
    connected_ues=[],
)

bs3 = BaseTower(
    id=3,
    latlng=LatLng(35.717580, -0.540077),  # center top
    connected_ues=[],
)


car = UserEquipment(
    id=100,
    latlng=LatLng(35.717122, -0.540052),  # Home
    serving_bs=bs2,  # starts connected to bs2
)
bs2.add_ue(ue=car)


rsrp1 = WaveUtils.calculate_rsrp(ue=car, bs=bs2)
print(f"rsrp1: {rsrp1:.2f} dBm")

car.move_meters(68, angle=270)


rsrp2 = WaveUtils.calculate_rsrp(ue=car, bs=bs2)
print(f"rsrp2: {rsrp2:.2f} dBm")


car.move_meters(56, angle=320)

rsrp3 = WaveUtils.calculate_rsrp(ue=car, bs=bs2)
print(f"rsrp3: {rsrp3:.2f} dBm")

car.move_meters(200, angle=270)


rsrp4 = WaveUtils.calculate_rsrp(ue=car, bs=bs2)
print(f"rsrp4: {rsrp4:.2f} dBm")

print(car.generate_report(all_bs=[bs1, bs2, bs3]))
