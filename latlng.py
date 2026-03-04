class LatLng:

    def __init__(self, lat: float, long: float):
        self.lat: float = lat
        self.long: float = long

    def __repr__(self):
        return f"LatLng(lat: {self.lat}, long: {self.long})"

    def __str__(self):
        return f"LatLng(lat: {self.lat}, long: {self.long})"
