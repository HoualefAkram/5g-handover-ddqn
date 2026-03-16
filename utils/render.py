import folium
from data_models.base_tower import BaseTower
from data_models.user_equipment import UserEquipment
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)


_UE_COLORS = ["blue", "purple", "orange", "darkred", "cadetblue", "darkgreen", "pink"]


class Render:
    @staticmethod
    def render_map(
        bs_list: list[BaseTower],
        ue_list: list[UserEquipment] | UserEquipment,
        output: str = "outputs/folium/simulation.html",
    ):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(ue_list, UserEquipment):
            ue_list = [ue_list]

        m = folium.Map(
            location=[bs_list[0].latlng.lat, bs_list[0].latlng.long], zoom_start=16
        )

        # Base Towers
        for bs in bs_list:
            folium.Marker(
                location=[bs.latlng.lat, bs.latlng.long],
                tooltip=f"BS {bs.id}",
                popup=f"BS{bs.id} | {bs.frequency/1e9:.1f}GHz | {bs.p_tx}dBm",
                icon=folium.Icon(color="black", icon="tower-cell", prefix="fa"),
            ).add_to(m)

        for i, ue in enumerate(ue_list):
            color = _UE_COLORS[i % len(_UE_COLORS)]

            # UE path
            if len(ue.path_history) > 1:
                folium.PolyLine(
                    locations=[[p.lat, p.long] for p in ue.path_history],
                    color=color,
                    weight=2,
                ).add_to(m)

            # UE initial position
            if ue.path_history:
                folium.Marker(
                    location=[ue.path_history[0].lat, ue.path_history[0].long],
                    tooltip=f"UE {ue.id} (start)",
                    icon=folium.Icon(color=color, icon="car", prefix="fa"),
                ).add_to(m)

            # UE current position
            if ue.latlng:
                folium.Marker(
                    location=[ue.latlng.lat, ue.latlng.long],
                    tooltip=f"UE {ue.id} (end)",
                    icon=folium.Icon(color=color, icon="car", prefix="fa"),
                ).add_to(m)

        m.save(output)
        print(Fore.GREEN + Style.BRIGHT + f"Map rendered and saved to {output}")
