import numpy as np
from latlng import LatLng


class LocationUtils:
    __earth_radius_meters = 6371000.0

    @staticmethod
    def haversine(pointA: LatLng, pointB: LatLng) -> float:
        """Distance between point A and point B. Result is in meters"""

        lat1: float = pointA.lat
        lon1: float = pointA.long

        lat2: float = pointB.lat
        lon2: float = pointB.long

        # Radius of the Earth in meters. Use 3958.8 for miles.
        R = LocationUtils.__earth_radius_meters

        # Convert latitude and longitude from degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Difference in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # The Haversine formula
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

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
        R = LocationUtils.__earth_radius_meters

        # Convert current latitude, longitude, and angle to radians
        lat1_rad = np.radians(point.lat)
        lon1_rad = np.radians(point.long)
        angle_rad = np.radians(angle)

        # Calculate the angular distance (distance divided by radius)
        ad = distance / R

        # Calculate new latitude
        lat2_rad = np.arcsin(
            np.sin(lat1_rad) * np.cos(ad)
            + np.cos(lat1_rad) * np.sin(ad) * np.cos(angle_rad)
        )

        # Calculate new longitude
        lon2_rad = lon1_rad + np.arctan2(
            np.sin(angle_rad) * np.sin(ad) * np.cos(lat1_rad),
            np.cos(ad) - np.sin(lat1_rad) * np.sin(lat2_rad),
        )

        # Update the car's state back in degrees
        latitude = np.degrees(lat2_rad)

        # Normalize longitude to stay between -180 and +180
        longitude = (np.degrees(lon2_rad) + 540) % 360 - 180

        return LatLng(lat=latitude, long=longitude)
