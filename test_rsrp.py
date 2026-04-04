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
from pathlib import Path
from utils.logger import Logger

# --- Params ---

SHOW_FOLIUM_OUTPUT = True
SHOW_TENSORBOARD_OUTPUT = True
FOLIUM_OUTPUT = "outputs/folium/simulation.html"
LOGDIR = "outputs/runs"
TEST_A3_RSRP = True
TEST_DDQN = True
SEED_COUNT = 10
SIMULATION_TIME = 900
STEP_LENGTH = 0.1

# --- Execution ---


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
            rsrp = report.rsrp_values.get(car.serving_bs.id, 0)
            car_handovers = car.get_total_handovers()
            car_pingpongs = car.get_total_pingpong()
            car_pingpongs_rate = car.get_pingpong_rate()
            car_rlf = car.rlf_count
            car_dho_time = car.dho_time

            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.RSRP,
                step=i,
                value=rsrp,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_HANDOVERS,
                step=i,
                value=car_handovers,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_PINGPONG,
                step=i,
                value=car_pingpongs,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.PINGPONG_RATE,
                step=i,
                value=car_pingpongs_rate,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_RLF,
                step=i,
                value=car_rlf,
            )
            logger.log_ue_metric(
                ue_index=car.id,
                metric=Logger.Metric.TOTAL_DHO,
                step=i,
                value=car_dho_time,
            )


    print()

    # Log final summary
    total_handovers = car.get_total_handovers()
    total_pingpong = car.get_total_pingpong()
    pingpong_rate = (
        total_pingpong / total_handovers if total_handovers > 0 else 0.0
    )

    print(Fore.RED + Style.BRIGHT + f"  Handovers: {total_handovers}")
    print(Fore.RED + Style.BRIGHT + f"  Ping Pongs: {total_pingpong}")
    print(
        Fore.RED + Style.BRIGHT + f"  Ping Pong rate: {pingpong_rate * 100:.2f}%"
    )

    return {
        "handovers": total_handovers,
        "pingpongs": total_pingpong,
        "pingpong_rate": pingpong_rate,
        "rlf": car.rlf_count,
        "dho": car.dho_time,
    }


if __name__ == "__main__":
    init(autoreset=True)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(Fore.CYAN + Style.BRIGHT + f"--- Starting RSRP Test ({SEED_COUNT} seeds) ---")

    # Base Stations (from test location: London 51.513377,-0.158129 to 51.493742,-0.141296)
    bs_list: list[BaseTower] = TowerDownloader.get_towers_from_cache()

    if not bs_list:
        error_text = (
            Fore.RED
            + Style.BRIGHT
            + "Error: No base stations found in this area. Exiting."
        )
        print(error_text)
        raise Exception(error_text)

    # Load DDQN model once if needed
    if TEST_DDQN:
        UserEquipment.load_model(
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        )

    all_results = {}

    # Performance loggers: aggregate avg/total across all seeds
    if TEST_A3_RSRP:
        a3_perf_logger = Logger(logdir=LOGDIR, name=f"PERF_A3_RSRP_LONDON_{timestamp}")
    if TEST_DDQN:
        ddqn_perf_logger = Logger(logdir=LOGDIR, name=f"PERF_DDQN_LONDON_{timestamp}")

    seed_pad = len(str(SEED_COUNT))

    for seed_idx in range(SEED_COUNT):
        seed = seed_idx + 1
        seed_label = str(seed).zfill(seed_pad)
        print()
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + f"{'='*60}"
        )
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + f"  Iteration {seed_idx + 1}/{SEED_COUNT} — SEED {seed}"
        )
        print(
            Fore.YELLOW
            + Style.BRIGHT
            + f"{'='*60}"
        )

        # Generate new route with this seed
        generate_trace(seed)
        fcd_data: list[dict[int, CarFcdData]] = FcdParser.parse_fcd_trace()

        iteration_results = {}

        # ===========================
        # A3 RSRP
        # ===========================
        if TEST_A3_RSRP:
            run_name = f"SEED{seed_label}_A3_RSRP_LONDON_{timestamp}"
            a3_rsrp_logger = Logger(logdir=LOGDIR, name=run_name)

            a3_rsrp_car = UserEquipment(
                id=0,
                all_bs=bs_list,
                print_logs_on_movement=False,
                handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
            )

            print(
                Fore.CYAN
                + Style.BRIGHT
                + f"  [{run_name}] Simulating A3 RSRP..."
            )

            result = simulation(
                bs_list=bs_list,
                fcd_data=fcd_data,
                logger=a3_rsrp_logger,
                car=a3_rsrp_car,
            )
            iteration_results["A3_RSRP"] = result
            a3_rsrp_logger.close()

        # ===========================
        # DDQN
        # ===========================
        if TEST_DDQN:
            # Wipe tower memory and fading state
            for bs in bs_list:
                bs.connected_ues.clear()
            WaveUtils.reset_fading_state()

            run_name = f"SEED{seed_label}_DDQN_LONDON_{timestamp}"
            ddqn_logger = Logger(logdir=LOGDIR, name=run_name)

            ddqn_car = UserEquipment(
                id=0,
                all_bs=bs_list,
                print_logs_on_movement=False,
                handover_algorithm=HandoverAlgorithm.DDQN,
            )

            print(
                Fore.CYAN
                + Style.BRIGHT
                + f"  [{run_name}] Simulating DDQN..."
            )

            result = simulation(
                bs_list=bs_list,
                fcd_data=fcd_data,
                logger=ddqn_logger,
                car=ddqn_car,
            )
            iteration_results["DDQN"] = result
            ddqn_logger.close()

        # Reset tower state between seeds
        for bs in bs_list:
            bs.connected_ues.clear()
        WaveUtils.reset_fading_state()

        all_results[seed] = iteration_results

        # Log running Performance (avg/total across seeds so far)
        step = seed_idx + 1
        completed_seeds = step

        if TEST_A3_RSRP and "A3_RSRP" in iteration_results:
            a3_vals = [all_results[s]["A3_RSRP"] for s in all_results if "A3_RSRP" in all_results[s]]
            total_ho = sum(v["handovers"] for v in a3_vals)
            total_pp = sum(v["pingpongs"] for v in a3_vals)
            total_rlf = sum(v["rlf"] for v in a3_vals)
            total_dho = sum(v["dho"] for v in a3_vals)
            a3_perf_logger.log_global_metric(Logger.Metric.TOTAL_HANDOVERS, total_ho, step)
            a3_perf_logger.log_global_metric(Logger.Metric.AVERAGE_HANDOVERS, total_ho / completed_seeds, step)
            a3_perf_logger.log_global_metric(Logger.Metric.TOTAL_PINGPONG, total_pp, step)
            a3_perf_logger.log_global_metric(Logger.Metric.AVERAGE_PINGPONG, total_pp / completed_seeds, step)
            a3_perf_logger.log_global_metric(Logger.Metric.PINGPONG_RATE, total_pp / total_ho if total_ho > 0 else 0, step)
            a3_perf_logger.log_global_metric(Logger.Metric.TOTAL_RLF, total_rlf, step)
            a3_perf_logger.log_global_metric(Logger.Metric.AVERAGE_RLF, total_rlf / completed_seeds, step)
            a3_perf_logger.log_global_metric(Logger.Metric.TOTAL_DHO, total_dho, step)
            a3_perf_logger.log_global_metric(Logger.Metric.AVERAGE_DHO, total_dho / completed_seeds, step)

        if TEST_DDQN and "DDQN" in iteration_results:
            ddqn_vals = [all_results[s]["DDQN"] for s in all_results if "DDQN" in all_results[s]]
            total_ho = sum(v["handovers"] for v in ddqn_vals)
            total_pp = sum(v["pingpongs"] for v in ddqn_vals)
            total_rlf = sum(v["rlf"] for v in ddqn_vals)
            total_dho = sum(v["dho"] for v in ddqn_vals)
            ddqn_perf_logger.log_global_metric(Logger.Metric.TOTAL_HANDOVERS, total_ho, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.AVERAGE_HANDOVERS, total_ho / completed_seeds, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.TOTAL_PINGPONG, total_pp, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.AVERAGE_PINGPONG, total_pp / completed_seeds, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.PINGPONG_RATE, total_pp / total_ho if total_ho > 0 else 0, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.TOTAL_RLF, total_rlf, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.AVERAGE_RLF, total_rlf / completed_seeds, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.TOTAL_DHO, total_dho, step)
            ddqn_perf_logger.log_global_metric(Logger.Metric.AVERAGE_DHO, total_dho / completed_seeds, step)

    # Close performance loggers
    if TEST_A3_RSRP:
        a3_perf_logger.close()
    if TEST_DDQN:
        ddqn_perf_logger.close()

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

    a3_totals = {"handovers": 0, "pingpongs": 0, "rlf": 0, "dho": 0}
    ddqn_totals = {"handovers": 0, "pingpongs": 0, "rlf": 0, "dho": 0}

    for seed, results in all_results.items():
        for algo, data in results.items():
            pp_rate_str = f"{data['pingpong_rate'] * 100:.1f}%"
            row = f"{seed:<6} | {algo:<10} | {data['handovers']:>10} | {data['pingpongs']:>10} | {pp_rate_str:>10} | {data['rlf']:>6} | {data['dho']:>8.2f}"
            color = Fore.MAGENTA if algo == "A3_RSRP" else Fore.CYAN
            print(color + row)

            if algo == "A3_RSRP":
                for k in a3_totals:
                    a3_totals[k] += data[k]
            else:
                for k in ddqn_totals:
                    ddqn_totals[k] += data[k]

    print(Fore.WHITE + "-" * len(header))

    if TEST_A3_RSRP:
        avg_ho = a3_totals["handovers"] / SEED_COUNT
        avg_pp = a3_totals["pingpongs"] / SEED_COUNT
        avg_rlf = a3_totals["rlf"] / SEED_COUNT
        avg_dho = a3_totals["dho"] / SEED_COUNT
        pp_rate = a3_totals["pingpongs"] / a3_totals["handovers"] if a3_totals["handovers"] > 0 else 0
        print(
            Fore.MAGENTA
            + Style.BRIGHT
            + f"{'AVG':<6} | {'A3_RSRP':<10} | {avg_ho:>10.1f} | {avg_pp:>10.1f} | {pp_rate * 100:>9.1f}% | {avg_rlf:>6.1f} | {avg_dho:>8.2f}"
        )

    if TEST_DDQN:
        avg_ho = ddqn_totals["handovers"] / SEED_COUNT
        avg_pp = ddqn_totals["pingpongs"] / SEED_COUNT
        avg_rlf = ddqn_totals["rlf"] / SEED_COUNT
        avg_dho = ddqn_totals["dho"] / SEED_COUNT
        pp_rate = ddqn_totals["pingpongs"] / ddqn_totals["handovers"] if ddqn_totals["handovers"] > 0 else 0
        print(
            Fore.CYAN
            + Style.BRIGHT
            + f"{'AVG':<6} | {'DDQN':<10} | {avg_ho:>10.1f} | {avg_pp:>10.1f} | {pp_rate * 100:>9.1f}% | {avg_rlf:>6.1f} | {avg_dho:>8.2f}"
        )

    print()

    # ===========================
    # Folium & TensorBoard Outputs
    # ===========================
    if SHOW_FOLIUM_OUTPUT:
        print(Fore.CYAN + Style.BRIGHT + "--- Rendering Final Output ---")
        last_car = ddqn_car if TEST_DDQN else (a3_rsrp_car if TEST_A3_RSRP else None)
        if last_car is not None:
            Render.render_map(bs_list=bs_list, ue_list=[last_car])
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
