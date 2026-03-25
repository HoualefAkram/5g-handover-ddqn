from data_models.car_fcd_data import CarFcdData
from data_models.handover_algorithm import HandoverAlgorithm
from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from utils.tower_downloader import TowerDownloader
from utils.render import Render
from utils.fcd_parser import FcdParser
from colorama import Fore, Style, init
import webbrowser
import subprocess
import time
from pathlib import Path
from utils.logger import Logger


# --- Params ---

SHOW_FOLIUM_OUTPUT = True
SHOW_TENSORBOARD_OUTPUT = True
FOLIUM_OUTPUT = "outputs/folium/simulation.html"
LOGDIR = "outputs/runs"

# --- Execution ---


init(autoreset=True)

logger = Logger(logdir=LOGDIR)
print(Fore.CYAN + Style.BRIGHT + f"--- Starting Test ---")

# Base Stations
bs_list: list[BaseTower] = TowerDownloader.get_towers_from_cache()

if not bs_list:
    error_text = (
        Fore.RED + Style.BRIGHT + "Error: No base stations found in this area. Exiting."
    )
    print(error_text)
    raise Exception(error_text)

# Parse Trace and Move Cars
fcd_data: list[dict[int, CarFcdData]] = FcdParser.parse_fcd_trace()


# Initialize User Equipment (Cars)
num_ue = FcdParser.count_vehicles()
cars: dict[int, UserEquipment] = {
    i: UserEquipment(
        id=i,
        all_bs=bs_list,
        print_logs_on_movement=False,
        handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
    )
    for i in range(num_ue)
}

print(
    Fore.CYAN
    + Style.BRIGHT
    + f"--- Simulating Movement and Network Logic for {num_ue} Vehicles ---"
)
total_steps = len(fcd_data)
start_time = time.time()

for i in range(total_steps):
    fcd = fcd_data[i]

    # print
    percent = (i / total_steps) * 100 if total_steps > 0 else 100

    elapsed_seconds = int(time.time() - start_time)
    mins, secs = divmod(elapsed_seconds, 60)
    timer_str = f"{mins:02d}:{secs:02d}"
    print(
        f"\r{Fore.CYAN}{Style.BRIGHT}{percent:.0f}% ,{i}/{total_steps} timesteps [Elapsed: {timer_str}]",
        flush=True,
        end="",
    )
    for car_id, car_data in fcd.items():
        if car_id in cars:  # Safe check in case SUMO spawned extra vehicles
            car = cars[car_id]
            report = car.move_to(car_data.latlng, timestep=car_data.timestep)
            rsrp = report.rsrp_values[car.serving_bs.id]
            rsrq = report.rsrq_values[car.serving_bs.id]
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.RSRP,
                step=car_data.timestep,
                value=rsrp,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.RSRQ,
                step=car_data.timestep,
                value=rsrq,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_HANDOVERS,
                step=car_data.timestep,
                value=car.get_total_handovers(),
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_PINGPONG,
                step=car_data.timestep,
                value=car.get_total_pingpong(),
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.PINGPONG_RATE,
                step=car_data.timestep,
                value=car.get_pingpong_rate(),
            )
print()

# Log global handover summary
last_timestep = FcdParser.last_timestep()
global_total_handovers = sum(ue.get_total_handovers() for ue in cars.values())
global_total_pingpong = sum(ue.get_total_pingpong() for ue in cars.values())
global_pingpong_rate = (
    global_total_pingpong / global_total_handovers
    if global_total_handovers > 0
    else 0.0
)
logger.log_global_metric(
    metric=Logger.Metric.TOTAL_HANDOVERS,
    value=global_total_handovers,
    step=last_timestep,
)
logger.log_global_metric(
    metric=Logger.Metric.TOTAL_PINGPONG,
    value=global_total_pingpong,
    step=last_timestep,
)
logger.log_global_metric(
    metric=Logger.Metric.PINGPONG_RATE,
    value=global_pingpong_rate,
    step=last_timestep,
)


for bs in bs_list:
    print(
        Fore.BLUE
        + f"Base Station {bs.id} served UEs: {[ue.id for ue in bs.connected_ues]}"
    )
print(Fore.RED + Style.BRIGHT + f"Global Handovers: {global_total_handovers}")
print(Fore.RED + Style.BRIGHT + f"Global Ping Pongs: {global_total_pingpong}")
print(Fore.RED + Style.BRIGHT + f"Global Ping Pong rate: {global_pingpong_rate}%")

# Render Final Map
print(Fore.CYAN + Style.BRIGHT + "--- Rendering Final Output ---")
Render.render_map(bs_list=bs_list, ue_list=list(cars.values()))
if SHOW_FOLIUM_OUTPUT:
    webbrowser.open(Path(FOLIUM_OUTPUT).resolve())

# Launch TensorBoard
logger.close()
time.sleep(1)
if SHOW_TENSORBOARD_OUTPUT:
    print(Fore.CYAN + Style.BRIGHT + "--- Launching TensorBoard ---")
    tb_port = 6006
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", LOGDIR, "--port", str(tb_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    webbrowser.open(f"http://localhost:{tb_port}")

    print(Fore.GREEN + Style.BRIGHT + "--- Test Done! ---")
    print(
        Fore.YELLOW
        + f"TensorBoard running at http://localhost:{tb_port} (PID: {tb_process.pid})"
    )
