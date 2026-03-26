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
        osm_file: str = "cache/maps/map.osm",
        network_output: str = "outputs/sumo/map.net.xml",
        trips_output: str = "outputs/sumo/trips.xml",
        route_output: str = "outputs/sumo/routes.xml",
        trace_output: str = "outputs/sumo/trace.xml",
        begin_simulation: float = 0,
        end_simulation: float = 900,
        spawn_interval: float = 3,
        step_length: float = 0.1,
        seed: int = 42,
        gui: bool = False,
        skip_netconvert: bool = False,
    ) -> None:
        self.osm_file = osm_file
        self.network = network_output
        self.route = route_output
        self.output = trace_output
        self.trips = trips_output
        self.begin = begin_simulation
        self.end = end_simulation
        self.step_length = step_length
        self.spawn_interval = spawn_interval
        self.seed = seed
        self.gui = gui
        self.skip_netconvert = skip_netconvert

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
            "-o",
            self.trips,
            "-p",
            str(self.spawn_interval),
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

        cmds = [netconverter_cmd, randomTrips_cmd, duarouter_cmd]
        if self.skip_netconvert:
            cmds.pop(0)
        return cmds

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
    def quick_run(
        osm_file: str = "cache/maps/map.osm",
        gui: bool = False,
        skip_netconvert: bool = False,
    ):
        PathGeneration(
            osm_file=osm_file,
            gui=gui,
            skip_netconvert=skip_netconvert,
            seed=randint(0, 10000),
        ).run()
