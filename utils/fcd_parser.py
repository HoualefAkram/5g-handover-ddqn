import xml.etree.ElementTree as ET
from pathlib import Path
from data_models.car_fcd_data import CarFcdData
from data_models.latlng import LatLng


class FcdParser:

    @staticmethod
    def parse_fcd_trace(
        trace_file: str = "outputs/sumo/trace.xml",
    ) -> list[dict[int, CarFcdData]]:
        if not Path(trace_file).exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        tree = ET.parse(trace_file)
        root = tree.getroot()

        timesteps = []
        for timestep in root.findall("timestep"):
            snapshot = {}
            for vehicle in timestep.findall("vehicle"):
                veh_id = int(vehicle.get("id"))
                lon = float(vehicle.get("x"))
                lat = float(vehicle.get("y"))
                time = float(timestep.get("time"))
                snapshot[veh_id] = CarFcdData(
                    id=veh_id,
                    latlng=LatLng(lat, lon),
                    timestep=time,
                )
            timesteps.append(snapshot)

        return timesteps

    @staticmethod
    def count_vehicles(
        trace_file: str = "outputs/sumo/trace.xml",
    ) -> int:
        if not Path(trace_file).exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        tree = ET.parse(trace_file)
        root = tree.getroot()

        vehicle_ids = set()
        for timestep in root.findall("timestep"):
            for vehicle in timestep.findall("vehicle"):
                vehicle_ids.add(int(vehicle.get("id")))

        return len(vehicle_ids)
