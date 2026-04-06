from data_models.car_fcd_data import CarFcdData
from data_models.handover_algorithm import HandoverAlgorithm
from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from utils.tower_downloader import TowerDownloader
from utils.render import Render
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
from collections import defaultdict
from utils.logger import Logger

# --- Params ---

SHOW_FOLIUM_OUTPUT = True
SHOW_TENSORBOARD_OUTPUT = True
FOLIUM_OUTPUT = "outputs/folium/simulation.html"
LOGDIR = "outputs/runs"
SEED = 42
SEED_COUNT = 10
SIMULATION_TIME = 900
STEP_LENGTH = 0.1

ALGORITHMS = [
    ("A3_RSRP", HandoverAlgorithm.A3_RSRP_3GPP, {}),
    ("DDQN", HandoverAlgorithm.DDQN, {}),
    ("DDQN_CHO", HandoverAlgorithm.DDQN_CHO, {}),
]

ALGO_COLORS = {
    "A3_RSRP": Fore.MAGENTA,
    "DDQN": Fore.CYAN,
    "DDQN_CHO": Fore.GREEN,
}

# --- Helpers ---


def generate_trace(seed: int):
    """Generate a new SUMO trace with the given seed."""
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
    """Run simulation for a single car (UE 0 only)."""
    total_steps = len(fcd_data)
    start_time = time.time()
    rsrp_per_step = {}

    for i in range(total_steps):
        fcd = fcd_data[i]

        # print progress
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        Fore.CYAN
        + Style.BRIGHT
        + f"--- Starting RSRP Test ({SEED_COUNT} seeds, {len(ALGORITHMS)} algorithms) ---"
    )

    # Base Stations (test location: London 51.513377,-0.158129 to 51.493742,-0.141296)
    bs_list: list[BaseTower] = TowerDownloader.get_towers_from_cache()

    if not bs_list:
        error_text = (
            Fore.RED
            + Style.BRIGHT
            + "Error: No base stations found in this area. Exiting."
        )
        print(error_text)
        raise Exception(error_text)

    # Load DDQN model once (shared by DDQN and DDQN_CHO)
    UserEquipment.load_model(
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Generate deterministic seeds from master SEED
    rng = random.Random(SEED)
    seeds = [rng.randint(0, 10000) for _ in range(SEED_COUNT)]
    seed_pad = len(str(SEED_COUNT))

    all_results = {}

    # PERF Loggers (1 per algorithm)
    perf_loggers = {}
    for algo_label, _, _ in ALGORITHMS:
        perf_loggers[algo_label] = Logger(
            logdir=LOGDIR, name=f"PERF_{algo_label}_LONDON_{timestamp}"
        )

    # ===========================
    # Seed Loop
    # ===========================
    for seed_idx in range(SEED_COUNT):
        seed = seeds[seed_idx]
        seed_label = str(seed_idx + 1).zfill(seed_pad)
        print()
        print(Fore.YELLOW + Style.BRIGHT + f"{'='*60}")
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + f"  Iteration {seed_idx + 1}/{SEED_COUNT} — SEED {seed}"
        )
        print(Fore.YELLOW + Style.BRIGHT + f"{'='*60}")

        # Generate new route with this seed
        generate_trace(seed)
        fcd_data: list[dict[int, CarFcdData]] = FcdParser.parse_fcd_trace()

        iteration_results = {}

        for algo_label, algo_enum, algo_kwargs in ALGORITHMS:
            # Reset tower state + fading before each algorithm
            for bs in bs_list:
                bs.connected_ues.clear()
            WaveUtils.reset_fading_state()

            # Per-seed per-algo logger
            run_name = f"SEED{seed_label}_{algo_label}_LONDON_{timestamp}"
            seed_logger = Logger(logdir=LOGDIR, name=run_name)

            car = UserEquipment(
                id=0,
                all_bs=bs_list,
                print_logs_on_movement=False,
                handover_algorithm=algo_enum,
                **algo_kwargs,
            )

            color = ALGO_COLORS.get(algo_label, Fore.WHITE)
            print(color + Style.BRIGHT + f"  [{run_name}] Simulating {algo_label}...")

            result = simulation(
                bs_list=bs_list,
                fcd_data=fcd_data,
                logger=seed_logger,
                car=car,
            )
            iteration_results[algo_label] = result
            seed_logger.close()

        # Reset tower state between seeds
        for bs in bs_list:
            bs.connected_ues.clear()
        WaveUtils.reset_fading_state()

        all_results[seed] = iteration_results

        # Log running PERF metrics (avg across seeds so far)
        step = seed_idx + 1
        completed_seeds = step

        for algo_label, _, _ in ALGORITHMS:
            if algo_label not in iteration_results:
                continue
            vals = [
                all_results[s][algo_label]
                for s in all_results
                if algo_label in all_results[s]
            ]
            avg_ho = sum(v["handovers"] for v in vals) / completed_seeds
            avg_pp = sum(v["pingpongs"] for v in vals) / completed_seeds
            total_ho = sum(v["handovers"] for v in vals)
            total_pp = sum(v["pingpongs"] for v in vals)
            avg_rlf = sum(v["rlf"] for v in vals) / completed_seeds
            avg_dho = sum(v["dho"] for v in vals) / completed_seeds
            perf = perf_loggers[algo_label]
            perf.log_global_metric(Logger.Metric.AVERAGE_HANDOVERS, avg_ho, step)
            perf.log_global_metric(Logger.Metric.AVERAGE_PINGPONG, avg_pp, step)
            perf.log_global_metric(
                Logger.Metric.PINGPONG_RATE,
                total_pp / total_ho if total_ho > 0 else 0,
                step,
            )
            perf.log_global_metric(Logger.Metric.AVERAGE_RLF, avg_rlf, step)
            perf.log_global_metric(Logger.Metric.AVERAGE_DHO, avg_dho, step)

    # ===========================
    # PERF: Average RSRP per step across seeds
    # ===========================
    for algo_label, _, _ in ALGORITHMS:
        rsrp_by_step = defaultdict(list)
        for s in all_results:
            if algo_label in all_results[s]:
                for step, rsrp in all_results[s][algo_label]["rsrp_per_step"].items():
                    rsrp_by_step[step].append(rsrp)
        for step in sorted(rsrp_by_step):
            avg_rsrp = sum(rsrp_by_step[step]) / len(rsrp_by_step[step])
            perf_loggers[algo_label].log_global_metric(
                Logger.Metric.AVERAGE_RSRP, avg_rsrp, step
            )

    # Close PERF loggers
    for algo_label, _, _ in ALGORITHMS:
        perf_loggers[algo_label].close()

    # ===========================
    # Final Summary
    # ===========================
    print()
    print(Fore.GREEN + Style.BRIGHT + f"{'='*60}")
    print(Fore.GREEN + Style.BRIGHT + f"  RESULTS SUMMARY ({SEED_COUNT} seeds)")
    print(Fore.GREEN + Style.BRIGHT + f"{'='*60}")

    header = f"{'Seed':<6} | {'Algorithm':<10} | {'Handovers':>10} | {'PingPongs':>10} | {'PP Rate':>10} | {'RLF':>6} | {'DHO':>8}"
    print(Fore.WHITE + Style.BRIGHT + header)
    print(Fore.WHITE + "-" * len(header))

    algo_totals = {
        label: {"handovers": 0, "pingpongs": 0, "rlf": 0, "dho": 0}
        for label, _, _ in ALGORITHMS
    }

    for seed, results in all_results.items():
        for algo, data in results.items():
            pp_rate_str = f"{data['pingpong_rate'] * 100:.1f}%"
            row = f"{seed:<6} | {algo:<10} | {data['handovers']:>10} | {data['pingpongs']:>10} | {pp_rate_str:>10} | {data['rlf']:>6} | {data['dho']:>8.2f}"
            color = ALGO_COLORS.get(algo, Fore.WHITE)
            print(color + row)

            for k in algo_totals[algo]:
                algo_totals[algo][k] += data[k]

    print(Fore.WHITE + "-" * len(header))

    for algo_label, _, _ in ALGORITHMS:
        t = algo_totals[algo_label]
        avg_ho = t["handovers"] / SEED_COUNT
        avg_pp = t["pingpongs"] / SEED_COUNT
        avg_rlf = t["rlf"] / SEED_COUNT
        avg_dho = t["dho"] / SEED_COUNT
        pp_rate = t["pingpongs"] / t["handovers"] if t["handovers"] > 0 else 0
        color = ALGO_COLORS.get(algo_label, Fore.WHITE)
        print(
            color
            + Style.BRIGHT
            + f"{'AVG':<6} | {algo_label:<10} | {avg_ho:>10.1f} | {avg_pp:>10.1f} | {pp_rate * 100:>9.1f}% | {avg_rlf:>6.1f} | {avg_dho:>8.2f}"
        )

    print()

    # ===========================
    # Folium & TensorBoard Outputs
    # ===========================
    if SHOW_FOLIUM_OUTPUT:
        print(Fore.CYAN + Style.BRIGHT + "--- Rendering Final Output ---")
        Render.render_map(bs_list=bs_list, ue_list=[car])
        webbrowser.open(Path(FOLIUM_OUTPUT).resolve().as_uri())

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
