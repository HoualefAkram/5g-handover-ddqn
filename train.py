from data_models.car_fcd_data import CarFcdData
from data_models.handover_algorithm import HandoverAlgorithm
from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from rl.handover_env import HandoverEnv
from utils.tower_downloader import TowerDownloader
from utils.render import Render
from utils.fcd_parser import FcdParser
from colorama import Fore, Style, init
from utils.logger import Logger


# --- Params ---
LOGDIR = "outputs/runs"
# --- Execution ---


init(autoreset=True)

logger = Logger(logdir=LOGDIR)
print(Fore.CYAN + Style.BRIGHT + f"--- Starting Training ---")

# Base Stations
bs_list: list[BaseTower] = TowerDownloader.get_towers_from_cache()

if not bs_list:
    error_text = (
        Fore.RED + Style.BRIGHT + "Error: No base stations found in this area. Exiting."
    )
    print(error_text)
    raise Exception(error_text)

# Initialize User Equipment (Cars)
num_ue = FcdParser.count_vehicles()
cars: dict[int, UserEquipment] = {
    i: UserEquipment(
        id=i,
        all_bs=bs_list,
        print_report_on_movement=False,
        handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
    )
    for i in range(num_ue)
}

env = HandoverEnv(
    base_towers=bs_list,
)
