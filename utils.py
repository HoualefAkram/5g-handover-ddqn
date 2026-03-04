import math
from latlng import LatLng


class Utils:
    __earth_radius_meters = 6371000.0

    @staticmethod
    def haversine(pointA: LatLng, pointB: LatLng) -> float:
        """Distance between point A and point B. Result is in meters"""

        lat1: float = pointA.lat
        lon1: float = pointA.long

        lat2: float = pointB.lat
        lon2: float = pointB.long

        # Radius of the Earth in meters. Use 3958.8 for miles.
        R = Utils.__earth_radius_meters

        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Difference in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # The Haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Calculate the final distance
        distance = R * c

        return distance

    @staticmethod
    def move_meters(point: LatLng, distance: float, angle: float) -> LatLng:
        """
        Moves specific distance (in meters) at a given angle (bearing in degrees).
        0 degrees is North, 90 is East, 180 is South, 270 is West.
        """
        # Earth's radius in meters
        R = Utils.__earth_radius_meters

        # Convert current latitude, longitude, and angle to radians
        lat1_rad = math.radians(point.lat)
        lon1_rad = math.radians(point.long)
        angle_rad = math.radians(angle)

        # Calculate the angular distance (distance divided by radius)
        ad = distance / R

        # Calculate new latitude
        lat2_rad = math.asin(
            math.sin(lat1_rad) * math.cos(ad)
            + math.cos(lat1_rad) * math.sin(ad) * math.cos(angle_rad)
        )

        # Calculate new longitude
        lon2_rad = lon1_rad + math.atan2(
            math.sin(angle_rad) * math.sin(ad) * math.cos(lat1_rad),
            math.cos(ad) - math.sin(lat1_rad) * math.sin(lat2_rad),
        )

        # Update the car's state back in degrees
        latitude = math.degrees(lat2_rad)

        # Normalize longitude to stay between -180 and +180
        longitude = (math.degrees(lon2_rad) + 540) % 360 - 180

        return LatLng(lat=latitude, long=longitude)
