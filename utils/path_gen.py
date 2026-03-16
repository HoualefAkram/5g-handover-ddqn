import subprocess
from pathlib import Path
import os
import sys
from random import randint
from colorama import Fore, Style, init

init(autoreset=True)


class PathGeneration:

    def __init__(
        self,
        osm_file: str = "maps/map.osm",
        network_output: str = "outputs/sumo/map.net.xml",
        trips_output: str = "outputs/sumo/trips.xml",
        route_output: str = "outputs/sumo/routes.xml",
        trace_output: str = "outputs/sumo/trace.xml",
        begin_simulation: float = 0,
        end_simulation: float = 3600,
        stop_trip_generation_after: float = 100,
        step_length: float = 1.0,
        seed: int = 42,
        gui: bool = False,
    ) -> None:
        self.osm_file = osm_file
        self.network = network_output
        self.route = route_output
        self.output = trace_output
        self.trips = trips_output
        self.begin = begin_simulation
        self.end = end_simulation
        self.stop_trip_generation_after = stop_trip_generation_after
        self.step_length = step_length
        self.seed = seed
        self.gui = gui

    def _validate_and_prepare(self) -> None:
        if not Path(self.osm_file).exists():
            raise FileNotFoundError(f"File not found: {self.osm_file}")

        Path(self.network).parent.mkdir(parents=True, exist_ok=True)
        Path(self.trips).parent.mkdir(parents=True, exist_ok=True)
        Path(self.route).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output).parent.mkdir(parents=True, exist_ok=True)

    def _build_simulation_scenario_cmds(self) -> list[list[str]]:
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise EnvironmentError(
                "SUMO_HOME environment variable is not set. Please ensure SUMO is installed correctly."
            )

        netconverter_exec = "netconvert"
        random_trips_exec = os.path.join(sumo_home, "tools", "randomTrips.py")
        duarouter_exec = "duarouter"

        netconverter_cmd = [
            netconverter_exec,
            "--osm-files",
            self.osm_file,
            "--output-file",
            self.network,
        ]
        randomTrips_cmd = [
            sys.executable,
            random_trips_exec,
            "-n",
            self.network,
            "-e",
            str(self.stop_trip_generation_after),
            "-o",
            self.trips,
            "--seed",
            str(self.seed),
            "--no-validate",
        ]
        duarouter_cmd = [
            duarouter_exec,
            "-n",
            self.network,
            "--route-files",
            self.trips,
            "-o",
            self.route,
            "--ignore-errors",
        ]

        return [netconverter_cmd, randomTrips_cmd, duarouter_cmd]

    def _build_generate_fcd_trace_cmd(self) -> list[str]:
        executable = "sumo-gui" if self.gui else "sumo"
        return [
            executable,
            "-n",
            self.network,
            "-r",
            self.route,
            "--fcd-output",
            self.output,
            "--fcd-output.geo",
            "true",
            "--begin",
            str(self.begin),
            "--end",
            str(self.end),
            "--step-length",
            str(self.step_length),
            "--seed",
            str(self.seed),
            "--quit-on-end",
            # "--no-step-log",
        ]

    def run(self) -> subprocess.CompletedProcess:
        print(Fore.CYAN + Style.BRIGHT + "--- Starting SUMO processes ---")
        self._validate_and_prepare()
        for cmd in self._build_simulation_scenario_cmds():
            cmd_str = " ".join(cmd)
            print(Fore.LIGHTCYAN_EX + f"Running: {cmd_str}")
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        cmd = self._build_generate_fcd_trace_cmd()
        cmd_str = " ".join(cmd)
        print(Fore.LIGHTCYAN_EX + f"Running: {cmd_str}")
        return subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    @staticmethod
    def quick_run(osm_file: str = "maps/map.osm", gui: bool = False):
        PathGeneration(osm_file=osm_file, gui=gui, seed=randint(0, 10000)).run()
