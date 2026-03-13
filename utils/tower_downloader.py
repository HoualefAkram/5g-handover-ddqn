import requests
from data_models.latlng import LatLng
from data_models.base_tower import BaseTower


class TowerDownloader:

    @staticmethod
    def get_towers_in_bbox(top_left: LatLng, bottom_right: LatLng) -> list[BaseTower]:
        min_lat = bottom_right.lat
        min_lon = top_left.long
        max_lat = top_left.lat
        max_lon = bottom_right.long

        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["telecom"="antenna"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["man_made"="mast"]["tower:type"="communication"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        """

        url = "https://overpass-api.de/api/interpreter"
        print("Fetching real-world cell towers from OpenStreetMap...")

        response = requests.post(url, data={"data": overpass_query})

        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to fetch towers. HTTP {response.status_code}"
            )

        data = response.json()
        elements = data.get("elements", [])

        towers = []
        for index, element in enumerate(elements):
            lat = element.get("lat")
            lon = element.get("lon")
            new_tower = BaseTower(
                id=index + 1,
                latlng=LatLng(lat=lat, long=lon),
                connected_ues=[],
            )
            towers.append(new_tower)

        print(f"Successfully found and initialized {len(towers)} BaseTowers!")
        return towers
