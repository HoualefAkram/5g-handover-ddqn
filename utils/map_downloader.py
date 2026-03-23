import requests
from pathlib import Path
from data_models.latlng import LatLng
from utils.osm_parser import OsmParser
from utils.location_utils import LocationUtils
import os
from colorama import Fore, Style, init

init(autoreset=True)


class MapDownloader:

    @staticmethod
    def download_osm_by_bbox(
        top_left: LatLng,
        bottom_right: LatLng,
        output_file: str = "cache/maps/map.osm",
    ) -> None:
        # left, bottom, right, top (min_lon, min_lat, max_lon, max_lat)
        min_lon = top_left.long
        min_lat = bottom_right.lat
        max_lon = bottom_right.long
        max_lat = top_left.lat

        # check if the map already exists
        if Path(output_file).exists():
            bounds = OsmParser.parse_bounds(output_file)
            if (
                LocationUtils.coords_are_identical(min_lon, bounds["minlon"])
                and LocationUtils.coords_are_identical(min_lat, bounds["minlat"])
                and LocationUtils.coords_are_identical(max_lon, bounds["maxlon"])
                and LocationUtils.coords_are_identical(max_lat, bounds["maxlat"])
            ):
                print(Fore.YELLOW + "Map already downloaded. Skipping...")
                return None

            print(Fore.RED + "Removing old map...")
            os.remove(output_file)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        url = f"https://overpass-api.de/api/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
        print(Fore.CYAN + f"Downloading OSM data from Overpass API to {output_file}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(Fore.GREEN + Style.BRIGHT + "Download complete!")
        else:
            error_text: str = (
                f"Failed to download map. HTTP Status: {response.status_code}\nResponse: {response.text}"
            )
            print(Fore.RED + Style.BRIGHT + error_text)
            raise Exception(error_text)
