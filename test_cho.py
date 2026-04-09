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
import random
from pathlib import Path
from datetime import datetime
from utils.logger import Logger

# --- Params ---

SHOW_TENSORBOARD_OUTPUT = True
LOGDIR = "outputs/runs"
SEED = 42
SEED_COUNT = 5
SIMULATION_TIME = 900
STEP_LENGTH = 0.1

# CHO confidence-gated sweep:
# confidence_threshold: softmax P(best) above which DDQN is trusted (no similarity)
# similarity_weight / q_weight: 50/50 fixed for the tiebreaker
CONFIDENCE_THRESHOLDS = [
    0.55,
    0.58,
    0.60,
    0.63,
    0.65,
    0.68,
    0.70,
    0.75,
]


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
    }


if __name__ == "__main__":
    init(autoreset=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        Fore.CYAN
        + Style.BRIGHT
        + f"--- Starting CHO Confidence Sweep ({SEED_COUNT} seeds) ---"
    )
    print(Fore.YELLOW + f"  Master SEED: {SEED}")
    print(Fore.YELLOW + f"  Confidence thresholds: {CONFIDENCE_THRESHOLDS}")

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

    # Generate deterministic seeds from master SEED
    rng = random.Random(SEED)
    seeds = [rng.randint(0, 10000) for _ in range(SEED_COUNT)]
    seed_pad = len(str(SEED_COUNT))

    # Labels: DDQN baseline + all CHO confidence variants
    algo_labels = ["DDQN"] + [f"CHO_ct{ct}" for ct in CONFIDENCE_THRESHOLDS]

    # all_results[label] = list of result dicts (one per seed)
    all_results = {label: [] for label in algo_labels}

    # PERF loggers (1 per algorithm variant)
    perf_loggers = {}
    for label in algo_labels:
        perf_loggers[label] = Logger(logdir=LOGDIR, name=f"PERF_{label}_{timestamp}")

    # ===========================
    # Seed Loop
    # ===========================
    for seed_idx in range(SEED_COUNT):
        seed = seeds[seed_idx]
        seed_label = str(seed_idx + 1).zfill(seed_pad)
        print()
        print(Fore.YELLOW + Style.BRIGHT + f"{'='*70}")
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + f"  Iteration {seed_idx + 1}/{SEED_COUNT} — SEED {seed}"
        )
        print(Fore.YELLOW + Style.BRIGHT + f"{'='*70}")

        # Generate new route with this seed
        generate_trace(seed)
        fcd_data: list[dict[int, CarFcdData]] = FcdParser.parse_fcd_trace()

        # --- Pure DDQN (baseline) ---
        for bs in bs_list:
            bs.connected_ues.clear()
        WaveUtils.reset_fading_state()

        run_name = f"SEED{seed_label}_DDQN_{timestamp}"
        ddqn_logger = Logger(logdir=LOGDIR, name=run_name)

        ddqn_car = UserEquipment(
            id=0,
            all_bs=bs_list,
            print_logs_on_movement=False,
            handover_algorithm=HandoverAlgorithm.DDQN,
        )

        print(Fore.MAGENTA + Style.BRIGHT + f"  [{run_name}] Simulating DDQN...")

        all_results["DDQN"].append(
            simulation(
                bs_list=bs_list,
                fcd_data=fcd_data,
                logger=ddqn_logger,
                car=ddqn_car,
            )
        )
        ddqn_logger.close()

        # --- DDQN_CHO with confidence threshold sweep ---
        for ct in CONFIDENCE_THRESHOLDS:
            label = f"CHO_ct{ct}"

            for bs in bs_list:
                bs.connected_ues.clear()
            WaveUtils.reset_fading_state()

            run_name = f"SEED{seed_label}_{label}_{timestamp}"
            cho_logger = Logger(logdir=LOGDIR, name=run_name)

            cho_car = UserEquipment(
                id=0,
                all_bs=bs_list,
                print_logs_on_movement=False,
                handover_algorithm=HandoverAlgorithm.DDQN_CHO,
                cho_confidence_threshold=ct,
                cho_similarity_weight=0.50,
                cho_q_weight=0.50,
            )

            print(
                Fore.CYAN
                + Style.BRIGHT
                + f"  [{run_name}] Simulating CHO (confidence_threshold={ct})..."
            )

            all_results[label].append(
                simulation(
                    bs_list=bs_list,
                    fcd_data=fcd_data,
                    logger=cho_logger,
                    car=cho_car,
                )
            )
            cho_logger.close()

        # Log running PERF metrics (avg across seeds so far)
        step = seed_idx + 1
        completed_seeds = step

        for label in algo_labels:
            runs = all_results[label]
            if len(runs) < completed_seeds:
                continue
            total_ho = sum(r["handovers"] for r in runs)
            total_pp = sum(r["pingpongs"] for r in runs)
            perf = perf_loggers[label]
            perf.log_global_metric(
                Logger.Metric.AVERAGE_HANDOVERS, total_ho / completed_seeds, step
            )
            perf.log_global_metric(
                Logger.Metric.AVERAGE_PINGPONG, total_pp / completed_seeds, step
            )
            perf.log_global_metric(
                Logger.Metric.PINGPONG_RATE,
                total_pp / total_ho if total_ho > 0 else 0,
                step,
            )
            perf.log_global_metric(
                Logger.Metric.AVERAGE_RLF,
                sum(r["rlf"] for r in runs) / completed_seeds,
                step,
            )
            perf.log_global_metric(
                Logger.Metric.AVERAGE_DHO,
                sum(r["dho"] for r in runs) / completed_seeds,
                step,
            )

    # Close PERF loggers
    for label in algo_labels:
        perf_loggers[label].close()

    # ===========================
    # Average Results Across Seeds
    # ===========================
    avg_results = {}
    for label in algo_labels:
        runs = all_results[label]
        n = len(runs)
        total_ho = sum(r["handovers"] for r in runs)
        total_pp = sum(r["pingpongs"] for r in runs)
        avg_results[label] = {
            "avg_handovers": total_ho / n,
            "avg_pingpongs": total_pp / n,
            "avg_pingpong_rate": total_pp / total_ho if total_ho > 0 else 0.0,
            "avg_rlf": sum(r["rlf"] for r in runs) / n,
            "avg_dho": sum(r["dho"] for r in runs) / n,
        }

    # ===========================
    # Final Summary
    # ===========================
    print()
    print(Fore.GREEN + Style.BRIGHT + f"{'='*80}")
    print(
        Fore.GREEN
        + Style.BRIGHT
        + f"  AVERAGE RESULTS ({SEED_COUNT} seeds, master SEED={SEED})"
    )
    print(Fore.GREEN + Style.BRIGHT + f"{'='*80}")

    header = f"{'Algorithm':<20} | {'Avg HO':>8} | {'Avg PP':>8} | {'PP Rate':>10} | {'Avg RLF':>8} | {'Avg DHO':>8}"
    print(Fore.WHITE + Style.BRIGHT + header)
    print(Fore.WHITE + "-" * len(header))

    # Find best CHO variant (lowest avg pingpong rate among CHO variants)
    cho_labels = [l for l in algo_labels if l != "DDQN"]
    best_label = min(cho_labels, key=lambda l: avg_results[l]["avg_pingpong_rate"])

    for label in algo_labels:
        d = avg_results[label]
        pp_rate_str = f"{d['avg_pingpong_rate'] * 100:.1f}%"

        if label == "DDQN":
            color = Fore.MAGENTA
        elif label == best_label:
            color = Fore.GREEN
        else:
            color = Fore.CYAN

        marker = " <-- BEST" if label == best_label else ""
        print(
            color
            + f"{label:<20} | {d['avg_handovers']:>8.1f} | {d['avg_pingpongs']:>8.1f} | {pp_rate_str:>10} | {d['avg_rlf']:>8.1f} | {d['avg_dho']:>8.2f}"
            + Style.BRIGHT
            + marker
        )

    # Extract best threshold
    best_ct = float(best_label.split("_ct")[1])
    print()
    print(Fore.GREEN + Style.BRIGHT + f"{'='*80}")
    print(Fore.GREEN + Style.BRIGHT + f"  BEST: confidence_threshold={best_ct}")
    print(
        Fore.GREEN
        + Style.BRIGHT
        + f"  Avg Handovers: {avg_results[best_label]['avg_handovers']:.1f}"
        + f"  |  Avg PingPongs: {avg_results[best_label]['avg_pingpongs']:.1f}"
        + f"  |  PP Rate: {avg_results[best_label]['avg_pingpong_rate'] * 100:.1f}%"
    )
    print(Fore.GREEN + Style.BRIGHT + f"{'='*80}")

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
