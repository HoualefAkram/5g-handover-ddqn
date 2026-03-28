# City Car Simulator — 5G Handover Simulation

A Python-based simulator that models **User Equipment (UEs)** — cars — moving through a real city with real **Base Station Towers** fetched from OpenCellID. The simulator calculates real-time **RSRP** and **RSRQ** signal metrics using standard radio propagation models, implementing 3GPP A3 handover decision logic. Designed as a foundation for **AI / Reinforcement Learning** agents to optimize handover decisions in 5G networks.

---

## What is Handover?

In mobile networks, as a car moves through a city, it constantly measures signal strength from nearby base stations. When a neighboring BS provides a stronger signal than the current serving BS, the network triggers a **handover** — switching the UE's connection to the better tower.

This simulator models that process from first principles using real map data, real tower locations, and realistic vehicle movement.

---

## Features

- **Real city maps** — downloads street maps from OpenStreetMap via Overpass API, cached locally to avoid redundant downloads
- **Real tower data** — fetches live LTE/NR tower locations from OpenCellID, cached locally to avoid redundant API calls
- **Realistic vehicle movement** — uses SUMO (Simulation of Urban Mobility) to generate traffic on actual streets
- **3GPP-compliant signal model** — log-distance path loss, RSRP, RSRQ, thermal noise
- **3GPP A3 handover logic** — hysteresis-based handover decisions (3 dB margin) with Time-to-Trigger (TTT)
- **Multi-UE support** — simulate multiple cars simultaneously
- **Interactive map output** — Folium HTML visualization, auto-opened in browser
- **TensorBoard logging** — per-UE RSRP/RSRQ metrics tracked over time, with auto-launch support
- **DDQN training pipeline** — Gymnasium environment, Double DQN agent, experience replay, checkpointing, GPU support

---

## Project Structure

```
city-car-simulator/
│
├── prepare.py                  # Data preparation — downloads maps, towers, generates SUMO traffic
├── test.py                     # Main simulation — runs handover sim, logs metrics, renders map
│
├── data_models/
│   ├── user_equipment.py       # UE class (car / mobile device) with handover logic
│   ├── base_tower.py           # BaseTower class (cellular BS)
│   ├── latlng.py               # LatLng coordinate dataclass
│   ├── ng_ran_report.py        # Signal measurement report (UE → BS)
│   ├── car_fcd_data.py         # SUMO FCD trace data per vehicle
│   └── handover_algorithm.py   # Enum: A3_RSRP_3GPP, DDQN_CHO
│
├── rl/
│   ├── handover_env.py         # Gymnasium environment for handover decisions
│   ├── ddqn_agent.py           # DDQN training loop (main training entry point)
│   ├── replay_buffer.py        # Experience replay buffer with disk persistence
│   └── checkpoint_manager.py   # Model checkpointing (epoch, epsilon, networks, optimizer)
│
├── utils/
│   ├── wave_utils.py           # RSRP, RSRQ, RSSI calculations
│   ├── location_utils.py       # Haversine distance, move_meters, coord comparison
│   ├── path_gen.py             # SUMO traffic generation interface
│   ├── map_downloader.py       # OSM map downloader (Overpass API) with bbox cache
│   ├── osm_parser.py           # OSM file bounds parser
│   ├── tower_downloader.py     # OpenCellID tower fetcher with bbox cache
│   ├── render.py               # Folium map visualization
│   ├── fcd_parser.py           # SUMO FCD XML parser
│   └── logger.py               # TensorBoard logging (per-UE and global metrics)
│
├── cache/
│   ├── maps/                   # Cached OSM map files
│   └── towers/                 # Cached tower JSON data
│
├── outputs/
│   ├── sumo/                   # SUMO network, routes, FCD traces
│   ├── folium/                 # HTML visualization output
│   └── runs/                   # TensorBoard log directories (timestamped)
│
└── .env                        # API keys (not committed)
```

---

## Quick Start

### 1. Prerequisites

**Python dependencies:**
```bash
pip install numpy folium requests python-dotenv colorama torch gymnasium tensorboard
```

**SUMO (Simulation of Urban Mobility):**
- Download and install from [sumo.dlr.de](https://sumo.dlr.de/)
- Set the `SUMO_HOME` environment variable to your SUMO installation path

**OpenCellID API key:**
- Register at [opencellid.org](https://opencellid.org/) to get a free API key
- Create a `.env` file in the project root:
  ```
  OPEN_CELL_ID_API_KEY="your_api_key_here"
  ```

### 2. Prepare Data

```bash
python prepare.py
```

This will:
1. Download the London OSM street map → cached to `cache/maps/` (skipped if bbox matches)
2. Fetch real LTE/NR towers from OpenCellID → cached to `cache/towers/` (skipped if bbox matches)
3. Generate vehicle traffic using SUMO (netconvert → randomTrips → duarouter → simulation)

### 3. Run Simulation (3GPP A3 Baseline)

```bash
python test.py
```

This will:
1. Load base stations from cached tower data
2. Parse SUMO FCD traces for vehicle positions
3. Run the handover simulation with 3GPP A3 logic
4. Log per-UE RSRP/RSRQ metrics to TensorBoard
5. Render an interactive map to `outputs/folium/simulation.html`
6. Auto-launch TensorBoard for metric visualization

### 4. Train DDQN Agent

```bash
python -m rl.ddqn_agent
```

This will:
1. Initialize the Gymnasium handover environment (generates new SUMO traffic each episode)
2. Train a Double DQN agent with epsilon-greedy exploration
3. Log training metrics (reward, loss, Q-values, handovers, ping-pongs) to TensorBoard
4. Save checkpoints to `cache/training/` after each epoch
5. Export the final model to `outputs/final_ddqn_model.pth`

Set `USE_GPU = True` (default) in `rl/ddqn_agent.py` to train on CUDA if available, or `False` to force CPU.

> **Note:** The replay buffer (`cache/training/replay_buffer.pkl`) and checkpoint (`cache/training/ddqn_checkpoint.pth`) persist across runs for seamless resume. If you change the environment config (map, towers, observation shape), delete `cache/training/` before retraining to avoid stale data.

---

## Configuration

Simulation parameters are configured in `prepare.py` and `test.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAP_TOP_LEFT` | `(51.519411, -0.148076)` | NW corner of simulation area (London) |
| `MAP_BOTTOM_RIGHT` | `(51.499324, -0.109732)` | SE corner of simulation area |
| `MCC` | `234` | Mobile Country Code (UK) |
| `SEED` | `200` | Random seed for reproducible SUMO traffic |
| `STEP_LENGTH` | `1` | Simulation step length in seconds |
| `SHOW_FOLIUM_OUTPUT` | `True` | Auto-open HTML output in browser (`test.py`) |
| `SHOW_TENSORBOARD_OUTPUT` | `True` | Auto-launch TensorBoard (`test.py`) |

### Training Hyperparameters (`rl/ddqn_agent.py`)

| Parameter | Default | Description |
|---|---|---|
| `USE_GPU` | `True` | Use CUDA GPU if available, `False` to force CPU |
| `epoches` | `500` | Number of training episodes |
| `lr` | `1e-3` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `decay_val` | `0.99` | Epsilon decay multiplier per epoch |
| `min_epsilon` | `0.05` | Minimum exploration rate |
| `update_rate` | `100` | Target network hard update interval (training steps) |
| `batch_size` | `32` | Replay buffer sample size |

---

## Signal Model

### Path Loss (Log-Distance)

```
PL(d) = PL(d0) + 10·n·log10(d/d0)
```

| Parameter | Value | Description |
|---|---|---|
| `d0` | 1 m | Reference distance |
| `n` (LOS) | 2.0 | Path loss exponent — clear line of sight |
| `n` (NLOS) | 3.0 | Path loss exponent — urban obstructions |
| LOS threshold | 20 m | Distance under which LOS is assumed |

### RSRP

```
RSRP (dBm) = P_tx + G_tx + G_rx - PL(d)
```

| Parameter | Value | Description |
|---|---|---|
| `P_tx` | 43 dBm | BS transmit power (20 W, typical 5G macro) |
| `G_tx` | 15 dBi | BS sector antenna gain |
| `G_rx` | 0 dBi | UE omnidirectional antenna gain |

### RSRQ

```
RSRQ (dB) = 10·log10(N) + RSRP - RSSI
```

Where `RSSI` = sum of signals from all detected BSs + thermal noise, and `N` = number of resource blocks.

### Thermal Noise Floor

```
noise (dBm) = -174 + 10·log10(bandwidth_hz) + noise_figure_db
```

| Parameter | Value |
|---|---|
| Bandwidth | 100 MHz (5G sub-6 GHz) |
| Noise figure | 7 dB (typical UE) |
| Noise floor | ~-87 dBm |

---

## Signal Quality Reference

### RSRP
| Value | Quality |
|---|---|
| > -60 dBm | Excellent |
| -60 to -80 dBm | Good |
| -80 to -90 dBm | Medium |
| -90 to -100 dBm | Poor |
| < -100 dBm | Very bad |

### RSRQ
| Value | Quality |
|---|---|
| > -3 dB | Excellent |
| -3 to -10 dB | Good |
| -10 to -15 dB | Medium |
| < -20 dB | Poor |

---

## Handover Logic

Implements the **3GPP A3 event**: a handover is triggered when:

```
RSRP(neighbor) > RSRP(serving) + hysteresis
```

- Default hysteresis: **3 dB**
- Time-to-Trigger (TTT): **3 seconds** — condition must hold for the last 3 seconds of reports
- Initial connection: UE automatically attaches to the strongest available tower
- Decisions are evaluated at every simulation timestep

### Handover Algorithms

| Algorithm | Status | Description |
|---|---|---|
| `A3_RSRP_3GPP` | Implemented | Standard 3GPP A3 event with hysteresis and TTT |
| `DDQN_CHO` | In Progress | Deep Double Q-Network for learned handover optimization |

---

## Reinforcement Learning

The project includes a full DDQN training pipeline for learned handover optimization:

- **Gymnasium Environment** (`rl/handover_env.py`) — action space: choose 1 of top-4 base stations; observation: 4 normalized RSRP + 4 normalized RSRQ + 4 serving one-hot = 12 features
- **DDQN Agent** (`rl/ddqn_agent.py`) — Double DQN with experience replay, epsilon-greedy exploration, target network hard updates, and GPU support
- **Replay Buffer** (`rl/replay_buffer.py`) — persistent experience replay with disk save/load
- **Checkpoint Manager** (`rl/checkpoint_manager.py`) — saves/resumes training state (epoch, epsilon, networks, optimizer) with cross-device support
- **TensorBoard Logger** (`utils/logger.py`) — tracks per-UE signal metrics, episode reward, loss, Q-values, handovers, and ping-pong rate

### Reward Design

The reward uses a **counterfactual regret** framework — all signals are compared at the same physical position using the latest measurement report:

| Scenario | Reward |
|---|---|
| Handover executed | `rsrp(new_tower) - rsrp(old_tower) + rsrq(new_tower) - rsrq(old_tower) - penalty` |
| Stay on best tower | `0.0` (optimal — no regret) |
| Stay on suboptimal tower | `(rsrp(current) - rsrp(best)) + (rsrq(current) - rsrq(best))` (negative regret) |

---

## Roadmap

- [x] Log-distance path loss model
- [x] RSRP calculation (per BS)
- [x] RSRQ / RSSI calculation
- [x] UE movement via SUMO on real streets
- [x] Real tower data from OpenCellID
- [x] Real map data from OpenStreetMap
- [x] Interactive map visualization (Folium)
- [x] 3GPP A3 handover trigger (hysteresis + TTT)
- [x] Multiple UEs
- [x] TensorBoard metric logging
- [x] Separated data preparation and simulation scripts
- [x] DDQN agent with full training loop
- [x] Gymnasium handover environment (top-4 filtering, counterfactual reward)
- [x] Experience replay with disk persistence
- [x] Checkpoint save/resume
- [x] GPU support (CUDA)
- [x] Performance metrics (ping-pong rate, handover count)
- [ ] Shadowing / slow fading
- [ ] Trained DDQN agent evaluation vs 3GPP A3 baseline

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Signal calculations |
| `folium` | Interactive map visualization |
| `requests` | Overpass API + OpenCellID HTTP requests |
| `python-dotenv` | Load API key from `.env` |
| `colorama` | Colored terminal output |
| `torch` | DDQN neural network (PyTorch) |
| `gymnasium` | RL environment interface |
| `tensorboard` | Training and signal metric visualization |
| SUMO | Traffic simulation engine (external) |

---

## References

- 3GPP TR 38.901 — Channel model for frequencies from 0.5 to 100 GHz
- 3GPP TS 36.214 — LTE physical layer measurements (RSRP, RSRQ definitions)
- Rappaport, T.S. — *Wireless Communications: Principles and Practice*

---

## License

MIT License — free to use, modify, and distribute.
