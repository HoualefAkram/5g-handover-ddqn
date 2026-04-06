from data_models.car_fcd_data import CarFcdData
from data_models.handover_algorithm import HandoverAlgorithm
from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from utils.tower_downloader import TowerDownloader
from utils.fcd_parser import FcdParser
from utils.wave_utils import WaveUtils
from utils.path_gen import PathGeneration
from colorama import Fore, Style, init
import torch
import webbrowser
import subprocess
import time
from pathlib import Path
from utils.logger import Logger

# --- Params ---

SHOW_TENSORBOARD_OUTPUT = True
LOGDIR = "outputs/runs"
SEED = 42
SIMULATION_TIME = 900
STEP_LENGTH = 0.1

# CHO weight sweep: q_weight fixed at 1, similarity_weight as tiebreaker (low values)
Q_WEIGHT = 1
SIMILARITY_WEIGHTS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]


# --- Helpers ---


def generate_trace(seed: int):
    path_gen = PathGeneration(
        end_simulation=SIMULATION_TIME,
        step_length=STEP_LENGTH,
        seed=seed,
        spawn_interval=5,
        skip_netconvert=True,
    )
    path_gen.run()


def simulation(
    logger: Logger,
    fcd_data: list[dict[int, CarFcdData]],
    bs_list: list[BaseTower],
    car: UserEquipment,
):
    total_steps = len(fcd_data)
    start_time = time.time()
    rsrp_per_step = {}

    for i in range(total_steps):
        fcd = fcd_data[i]

        percent = (i / total_steps) * 100 if total_steps > 0 else 100
        elapsed_seconds = int(time.time() - start_time)
        mins, secs = divmod(elapsed_seconds, 60)
        timer_str = f"{mins:02d}:{secs:02d}"
        print(
            f"\r{Fore.CYAN}{Style.BRIGHT}{percent:.0f}% ,{i}/{total_steps} timesteps [Elapsed: {timer_str}]",
            flush=True,
            end="",
        )

        if car.id not in fcd:
            continue

        car_data = fcd[car.id]
        report = car.move_to(
            car_data.latlng,
            timestep=car_data.timestep,
            speed=car_data.speed,
            angle=car_data.angle,
        )

        if car.serving_bs:
            rsrp = WaveUtils.normalize_rsrp_index(
                rsrp_index=report.rsrp_values.get(car.serving_bs.id, 0),
                radio_type=car.serving_bs.radio,
            )
            rsrp_per_step[i] = rsrp

            logger.log_ue_metric(
                ue_index=car.id, metric=Logger.Metric.RSRP, step=i, value=rsrp
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_HANDOVERS,
                step=i,
                value=car.get_total_handovers(),
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_PINGPONG,
                step=i,
                value=car.get_total_pingpong(),
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.PINGPONG_RATE,
                step=i,
                value=car.get_pingpong_rate(),
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_RLF,
                step=i,
                value=car.rlf_count,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_DHO,
                step=i,
                value=car.dho_time,
            )

    print()

    total_handovers = car.get_total_handovers()
    total_pingpong = car.get_total_pingpong()
    pingpong_rate = total_pingpong / total_handovers if total_handovers > 0 else 0.0

    print(Fore.RED + Style.BRIGHT + f"  Handovers: {total_handovers}")
    print(Fore.RED + Style.BRIGHT + f"  Ping Pongs: {total_pingpong}")
    print(Fore.RED + Style.BRIGHT + f"  Ping Pong rate: {pingpong_rate * 100:.2f}%")

    return {
        "handovers": total_handovers,
        "pingpongs": total_pingpong,
        "pingpong_rate": pingpong_rate,
        "rlf": car.rlf_count,
        "dho": car.dho_time,
        "rsrp_per_step": rsrp_per_step,
    }


if __name__ == "__main__":
    init(autoreset=True)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        Fore.CYAN
        + Style.BRIGHT
        + f"--- Starting CHO Test (SEED {SEED}) ---"
    )
    print(Fore.YELLOW + f"  Q weight: {Q_WEIGHT} (fixed)")
    print(Fore.YELLOW + f"  Similarity weights: {SIMILARITY_WEIGHTS}")

    # Base Stations
    bs_list: list[BaseTower] = TowerDownloader.get_towers_from_cache()

    if not bs_list:
        error_text = (
            Fore.RED
            + Style.BRIGHT
            + "Error: No base stations found in this area. Exiting."
        )
        print(error_text)
        raise Exception(error_text)

    # Load DDQN model
    UserEquipment.load_model(
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Generate route once — same for all runs
    generate_trace(SEED)
    fcd_data: list[dict[int, CarFcdData]] = FcdParser.parse_fcd_trace()

    algo_labels = ["DDQN"] + [f"CHO_s{sw}_q{Q_WEIGHT}" for sw in SIMILARITY_WEIGHTS]
    results = {}

    # ===========================
    # Pure DDQN (baseline)
    # ===========================
    for bs in bs_list:
        bs.connected_ues.clear()
    WaveUtils.reset_fading_state()

    run_name = f"DDQN_{timestamp}"
    ddqn_logger = Logger(logdir=LOGDIR, name=run_name)

    ddqn_car = UserEquipment(
        id=0,
        all_bs=bs_list,
        print_logs_on_movement=False,
        handover_algorithm=HandoverAlgorithm.DDQN,
    )

    print(Fore.CYAN + Style.BRIGHT + f"  [{run_name}] Simulating DDQN...")

    results["DDQN"] = simulation(
        bs_list=bs_list,
        fcd_data=fcd_data,
        logger=ddqn_logger,
        car=ddqn_car,
    )
    ddqn_logger.close()

    # ===========================
    # DDQN_CHO with weight sweep
    # ===========================
    for sw in SIMILARITY_WEIGHTS:
        label = f"CHO_s{sw}_q{Q_WEIGHT}"

        for bs in bs_list:
            bs.connected_ues.clear()
        WaveUtils.reset_fading_state()

        run_name = f"{label}_{timestamp}"
        cho_logger = Logger(logdir=LOGDIR, name=run_name)

        cho_car = UserEquipment(
            id=0,
            all_bs=bs_list,
            print_logs_on_movement=False,
            handover_algorithm=HandoverAlgorithm.DDQN_CHO,
            cho_similarity_weight=sw,
            cho_q_weight=Q_WEIGHT,
        )

        print(
            Fore.CYAN
            + Style.BRIGHT
            + f"  [{run_name}] Simulating CHO (sim={sw}, q={Q_WEIGHT})..."
        )

        results[label] = simulation(
            bs_list=bs_list,
            fcd_data=fcd_data,
            logger=cho_logger,
            car=cho_car,
        )
        cho_logger.close()

    # ===========================
    # Final Summary
    # ===========================
    print()
    print(Fore.GREEN + Style.BRIGHT + f"{'='*70}")
    print(Fore.GREEN + Style.BRIGHT + f"  RESULTS SUMMARY (SEED {SEED})")
    print(Fore.GREEN + Style.BRIGHT + f"{'='*70}")

    header = f"{'Algorithm':<16} | {'Handovers':>10} | {'PingPongs':>10} | {'PP Rate':>10} | {'RLF':>6} | {'DHO':>8}"
    print(Fore.WHITE + Style.BRIGHT + header)
    print(Fore.WHITE + "-" * len(header))

    for label in algo_labels:
        data = results[label]
        pp_rate = data["pingpong_rate"] * 100
        color = Fore.MAGENTA if label == "DDQN" else Fore.CYAN
        print(
            color
            + f"{label:<16} | {data['handovers']:>10} | "
            f"{data['pingpongs']:>10} | {pp_rate:>9.1f}% | "
            f"{data['rlf']:>6} | {data['dho']:>8.2f}"
        )

    print()

    # ===========================
    # TensorBoard
    # ===========================
    if SHOW_TENSORBOARD_OUTPUT:
        print(Fore.CYAN + Style.BRIGHT + "--- Launching TensorBoard ---")
        tb_port = 6006
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", LOGDIR, "--port", str(tb_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)
        webbrowser.open(f"http://localhost:{tb_port}")

        print(Fore.GREEN + Style.BRIGHT + "--- Test Done! ---")
        print(
            Fore.YELLOW
            + f"TensorBoard running at http://localhost:{tb_port} (PID: {tb_process.pid})"
        )
