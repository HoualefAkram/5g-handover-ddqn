# 5G Handover DDQN — Learned Handover Optimization vs 3GPP A3 Baseline

A Python-based simulator that models **User Equipment (UEs)** — cars — moving through a real city with real **Base Station Towers** fetched from OpenCellID. The simulator calculates real-time **RSRP** and **RSRQ** signal metrics using standard radio propagation models and compares a **3GPP A3 handover baseline** against a **trained Double DQN agent** for optimized handover decisions in 5G networks.

---

## What is Handover?

In mobile networks, as a car moves through a city, it constantly measures signal strength from nearby base stations. When a neighboring BS provides a stronger signal than the current serving BS, the network triggers a **handover** — switching the UE's connection to the better tower.

This simulator models that process from first principles using real map data, real tower locations, and realistic vehicle movement.

---

## Features

- **Real city maps** — downloads street maps from OpenStreetMap via Overpass API, cached locally to avoid redundant downloads
- **Real tower data** — downloads the full OpenCellID CSV database per MCC, filters LTE/NR towers within the simulation bbox, cached locally to avoid redundant downloads
- **Realistic vehicle movement** — uses SUMO (Simulation of Urban Mobility) to generate traffic on actual streets
- **3GPP-compliant signal model** — log-distance path loss, shadow fading (Gudmundson), fast fading (Rician/Rayleigh), RSRP, RSRQ, thermal noise
- **3GPP A3 handover logic** — hysteresis-based handover decisions (2 dB margin) with Time-to-Trigger (TTT)
- **Multi-UE support** — simulate multiple cars simultaneously
- **Interactive map output** — Folium HTML visualization, auto-opened in browser
- **TensorBoard logging** — per-UE and system-level metrics (RSRP, RSRQ, handovers, ping-pongs, RLF, handover delay) with auto-launch support
- **DDQN training pipeline** — Gymnasium environment, Double DQN agent, experience replay, checkpointing, GPU support

---

## Project Structure

```
5g-handover-ddqn/
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
│   ├── q_network.py            # QNetwork (PyTorch nn.Module) — DDQN policy/target network
│   └── handover_algorithm.py   # Enum: A3_RSRP_3GPP, DDQN_CHO, DDQN, NONE
│
├── rl/
│   ├── handover_env.py         # Gymnasium environment for handover decisions
│   ├── ddqn_agent.py           # DDQN training loop (main training entry point)
│   ├── replay_buffer.py        # Experience replay buffer with disk persistence
│   └── checkpoint_manager.py   # Model checkpointing (epoch, epsilon, networks, optimizer)
│
├── helpers/
│   ├── filters.py              # Top-k tower filtering by normalized RSRP score
│   └── functions.py            # Softmax, cosine similarity, bearing, weighted sum
│
├── utils/
│   ├── wave_utils.py           # RSRP, RSRQ, RSSI, path loss, shadow/fast fading calculations
│   ├── location_utils.py       # Haversine distance, move_meters, coord comparison
│   ├── path_gen.py             # SUMO traffic generation interface
│   ├── map_downloader.py       # OSM map downloader (Overpass API) with bbox cache
│   ├── osm_parser.py           # OSM file bounds parser
│   ├── tower_downloader.py     # OpenCellID CSV downloader with local bbox filtering and cache
│   ├── render.py               # Folium map visualization
│   ├── fcd_parser.py           # SUMO FCD XML parser
│   └── logger.py               # TensorBoard logging (per-UE and global metrics)
│
├── test_rsrp.py                # A3 RSRP baseline testing with multiple seeds
├── test_cho.py                 # CHO weight sweep testing
│
├── architecture.md             # System architecture diagrams
├── sources.md                  # Technical references and design-choice justifications
│
├── cache/
│   ├── maps/                   # Cached OSM map files
│   ├── towers/                 # Cached tower JSON data
│   └── training/               # Model checkpoints & replay buffer
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
1. Download the OSM street map for the configured area → cached to `cache/maps/` (skipped if bbox matches)
2. Download the full OpenCellID CSV database for the configured MCC, then filter LTE/NR towers within the bbox → cached to `cache/towers/` (skipped if bbox matches)
3. Generate vehicle traffic using SUMO (netconvert → randomTrips → duarouter → simulation)

### 3. Run Simulation & Evaluation

```bash
python test.py
```

This will:
1. Load base stations from cached tower data
2. Parse SUMO FCD traces for vehicle positions
3. Run the 3GPP A3 RSRP baseline simulation (if `TEST_A3_RSRP = True`)
4. Run the DDQN handover simulation (if `TEST_DDQN = True`, requires a trained model at `outputs/final_ddqn_model.pth`)
5. Log per-UE and system-level metrics (RSRP, RSRQ, handovers, ping-pongs, averages) to TensorBoard
6. Render an interactive map to `outputs/folium/simulation.html`
7. Auto-launch TensorBoard for side-by-side comparison of A3 vs DDQN runs

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

### 5. Alternative Test Runners

```bash
python test_rsrp.py    # A3 RSRP baseline with multiple seeds (10 runs, 900s each)
python test_cho.py     # CHO weight sweep (similarity vs Q-value trade-off)
```

---

## Configuration

Simulation parameters are configured in `prepare.py` and `test.py`. The default area is in central London, UK:

| Parameter | Default | Description |
|---|---|---|
| `MAP_TOP_LEFT` | `(51.519480, -0.169511)` | NW corner of simulation area (London, UK) |
| `MAP_BOTTOM_RIGHT` | `(51.479214, -0.105529)` | SE corner of simulation area |
| `MCC` | `234` | Mobile Country Code (UK) |
| `SEED` | `100` | Random seed for reproducible SUMO traffic |
| `SIMULATION_TIME` | `300` | Simulation duration in seconds (5 minutes) |
| `STEP_LENGTH` | `0.1` | Simulation step length in seconds (100 ms) |
| `SPAWN_INTERVAL` | `5` | Vehicle spawn interval in seconds (SUMO randomTrips) |
| `SHOW_FOLIUM_OUTPUT` | `True` | Auto-open HTML output in browser (`test.py`) |
| `SHOW_TENSORBOARD_OUTPUT` | `True` | Auto-launch TensorBoard (`test.py`) |
| `TEST_A3_RSRP` | `True` | Run 3GPP A3 RSRP baseline simulation (`test.py`) |
| `TEST_DDQN` | `True` | Run DDQN handover simulation (`test.py`) |

### Training Hyperparameters (`rl/ddqn_agent.py`)

| Parameter | Default | Description |
|---|---|---|
| `USE_GPU` | `True` | Use CUDA GPU if available, `False` to force CPU |
| `episodes` | `800` | Number of training episodes |
| `lr` | `5e-4` | Adam learning rate |
| `gamma` | `0.97` | Discount factor |
| `decay_val` | `0.99` | Epsilon decay multiplier per episode |
| `min_epsilon` | `0.05` | Minimum exploration rate |
| `target_update_episodes` | `2` | Target network hard update interval (episodes) |
| `train_every` | `20` | Backprop frequency (every N environment steps) |
| `batch_size` | `128` | Replay buffer sample size |
| `min_buffer_size` | `1000` | Minimum experiences before training starts |

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
| `n` (NLOS) | 3.5 | Path loss exponent — dense urban obstructions (Rappaport range 2.7–3.5) |
| LOS threshold | 5 m | Distance under which LOS is assumed |

### RSRP

```
RSRP (dBm) = P_tx + G_tx + G_rx - PL(d) - L_shadow + L_fast
```

| Parameter | LTE | NR | Description |
|---|---|---|---|
| `P_tx` | 46 dBm (40W) | 43 dBm (20W) | BS transmit power (3GPP TS 36.104 / 38.104) |
| `G_tx` | 15 dBi | 17 dBi | BS antenna gain (MIMO beamforming for NR) |
| `G_rx` | 0 dBi | 0 dBi | UE omnidirectional antenna gain |
| Frequency | 1800 MHz (Band 3) | 3500 MHz (n78) | Carrier frequency |

### Shadow Fading (Gudmundson Model)

Spatially correlated log-normal fading per link:

```
S_new = r · S_old + sqrt(1 - r^2) · N(0, sigma)
r = exp(-d_moved / d_corr)
```

| Parameter | LOS | NLOS | Source |
|---|---|---|---|
| `sigma` | 6.0 dB | 8.93 dB | 3GPP TR 38.901 Table 7.5-6 InH-Office |
| `d_corr` | 10 m | 10 m | 3GPP TR 38.901 Table 7.6.3.1-2 InH |

### Fast Fading

| Condition | Model | K-factor | Source |
|---|---|---|---|
| LOS | Rician | 7 dB | 3GPP TR 38.901 Table 7.5-6 UMa-LOS |
| NLOS | Rayleigh | - | No dominant path component |

### RSRQ

```
RSRQ (dB) = RSRP - RSSI
```

Where `RSSI` = sum of signals from all detected BSs + thermal noise.

### Thermal Noise Floor

```
noise (dBm) = -174 + 10·log10(bandwidth_hz) + noise_figure_db
```

| Parameter | LTE | NR |
|---|---|---|
| Bandwidth | 20 MHz | 100 MHz |
| Noise figure | 7 dB | 7 dB |
| Noise floor | ~-100 dBm | ~-87 dBm |

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

- Default hysteresis: **2 dB**
- Time-to-Trigger (TTT): **160 ms** — condition must hold for the last 160 ms of reports
- Initial connection: UE automatically attaches to the strongest available tower
- Decisions are evaluated at every simulation timestep

### Radio Link Failure (RLF) Detection

A simplified RSRP-based proxy for the 3GPP Qout mechanism. In the standard (TS 38.133 Section 8.1.1 for NR, TS 36.133 Section 7.6 for LTE), Qout is defined as 10% BLER of a hypothetical PDCCH — an SINR/BLER threshold, not RSRP. This simulator uses a per-RAT normalized RSRP threshold mapped to approximately **-116 dBm** as a practical cell-edge proxy:

| RAT | Normalized Threshold | RSRP Equivalent |
|---|---|---|
| NR | `41/127 ≈ 0.323` | ≈ -116 dBm |
| LTE | `25/97 ≈ 0.258` | ≈ -116 dBm |

When the serving cell RSRP falls below the threshold at any timestep, the UE's RLF counter increments by 1.

### Handover Interruption Time (DHO)

Each handover adds a fixed interruption delay to the UE's accumulated handover delay cost (`dho_time`), based on 3GPP intra-frequency known-cell requirements:

| RAT | Interruption Time | Source |
|---|---|---|
| NR | 20 ms | TS 38.133 Section 8.2.2 |
| LTE | 40 ms | TS 36.133 Section 8.1.1.1 |

### Handover Algorithms

| Algorithm | Status | Description |
|---|---|---|
| `A3_RSRP_3GPP` | Implemented | Standard 3GPP A3 event with hysteresis and TTT |
| `DDQN_CHO` | Implemented | Deep Double Q-Network with direction-aware conditional handover (bearing/cosine similarity re-ranking) |
| `DDQN` | Implemented | Pure DDQN (argmax Q-values, no direction-aware post-processing) |
| `NONE` | — | No handover (stay on initial tower) |

---

## Reinforcement Learning

The project includes a full DDQN training pipeline for learned handover optimization:

- **QNetwork** (`data_models/q_network.py`) — PyTorch `nn.Module` (14 → 256 → 128 → 64 → 4) with GELU activations, hard target-network update, and `from_state_dict` factory
- **Gymnasium Environment** (`rl/handover_env.py`) — action space: choose 1 of top-4 base stations; observation: 4 normalized RSRP + 4 RSRP trend (delta from previous report) + 4 serving one-hot + 1 normalized speed + 1 normalized time since last handover = 14 features
- **DDQN Agent** (`rl/ddqn_agent.py`) — Double DQN with experience replay, epsilon-greedy exploration, target network hard updates, and GPU support
- **Replay Buffer** (`rl/replay_buffer.py`) — persistent experience replay with disk save/load
- **Checkpoint Manager** (`rl/checkpoint_manager.py`) — saves/resumes training state (epoch, epsilon, networks, optimizer) with cross-device support
- **Top-k Filtering** (`helpers/filters.py`) — selects the k best candidate towers by normalized RSRP score for the observation space
- **Helper Functions** (`helpers/functions.py`) — softmax, cosine similarity, bearing calculation, and weighted sum used by the DDQN handover logic
- **TensorBoard Logger** (`utils/logger.py`) — tracks per-UE metrics (RSRP, RSRQ, handovers, ping-pongs) and system-level metrics (averages, totals, ping-pong rate), plus training metrics (reward, loss, Q-values, epsilon)

### Reward Design

The reward uses a **counterfactual delta** framework — signals from the old and new tower are compared at the same physical position using the latest measurement report:

| Scenario | Reward |
|---|---|
| Handover executed | `rsrp(new_tower) - rsrp(old_tower) - dynamic_penalty` |
| Stay (no handover) | `rsrp(serving) - rsrp(best_alternative)` — positive if serving is stronger, negative if a better option exists |
| RLF penalty | `-1.5` applied on top of any reward when serving RSRP falls below the RLF threshold |

The "stay" reward uses a **counterfactual comparison**: the agent is rewarded for staying on a strong tower and penalized for ignoring a better alternative. Combined with the handover penalty, this creates a learned hysteresis that balances signal quality against switching cost.

#### Dynamic Handover Penalty (Cooldown)

The handover penalty scales based on time since the last handover, penalizing rapid switching more heavily:

```
penalty = cooldown_penalty * (1 - time_since_ho)
```

Where `time_since_ho` is normalized to `[0, 1]` based on `min_time_of_stay`.

| Parameter | Value | Description |
|---|---|---|
| `cooldown_penalty` | `0.5` | Maximum penalty (decays with time since last handover) |
| `min_time_of_stay` | `1.0 s` | Cooldown period before penalty reaches zero |

This produces a penalty range of `0.0` (fully cooled down) to `0.5` (immediate re-switch), directly discouraging ping-pong handovers while allowing justified handovers after the cooldown.

---

## TensorBoard Metrics

### Per-UE Metrics (tagged `UE_{id}/`)
| Metric | Description |
|---|---|
| `RSRP` | Serving cell RSRP (dBm) per timestep |
| `RSRQ` | Serving cell RSRQ (dB) per timestep |
| `TOTAL_HANDOVERS` | Cumulative handover count |
| `TOTAL_PINGPONG` | Cumulative ping-pong count |
| `PINGPONG_RATE` | Ping-pong / handover ratio |
| `TOTAL_RLF` | Cumulative radio link failure count |
| `TOTAL_DHO` | Cumulative handover interruption delay (seconds) |

### System-Level Metrics (tagged `Performance/`)
| Metric | Description |
|---|---|
| `AVERAGE_RSRP` | Mean RSRP across all active UEs |
| `AVERAGE_RSRQ` | Mean RSRQ across all active UEs |
| `TOTAL_HANDOVERS` | Sum of handovers across all UEs |
| `AVERAGE_HANDOVERS` | Mean handovers per UE |
| `TOTAL_PINGPONG` | Sum of ping-pongs across all UEs |
| `AVERAGE_PINGPONG` | Mean ping-pongs per UE |
| `PINGPONG_RATE` | Sum of per-UE ping-pong rates |
| `AVERAGE_PINGPONG_RATE` | Mean ping-pong rate per UE |
| `TOTAL_RLF` | Sum of RLF events across all UEs |
| `AVERAGE_RLF` | Mean RLF count per UE |
| `TOTAL_DHO` | Sum of handover delay across all UEs (seconds) |
| `AVERAGE_DHO` | Mean handover delay per UE (seconds) |

### Training Metrics (DDQN agent)
| Metric | Description |
|---|---|
| `EPISODE_LENGTH` | Steps per training episode |
| `TOTAL_REWARD` | Cumulative reward per episode |
| `AVERAGE_MAX_Q` | Mean max Q-value per episode |
| `AVERAGE_LOSS` | Mean Huber loss per episode |
| `EPSILON` | Current exploration rate |

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
- [x] Gymnasium handover environment (top-4 filtering, counterfactual reward, RLF penalty)
- [x] RSRP trend observation (delta between consecutive reports per tower)
- [x] Speed-aware observation space (network-side UE speed estimation)
- [x] Time-since-handover observation feature (handover cooldown awareness)
- [x] Dynamic handover penalty (cooldown-based, anti-ping-pong)
- [x] Experience replay with disk persistence
- [x] Checkpoint save/resume
- [x] GPU support (CUDA)
- [x] Performance metrics (ping-pong rate, handover count, RLF, handover delay)
- [x] Shadow fading (Gudmundson correlated log-normal)
- [x] Fast fading (Rician LOS / Rayleigh NLOS)
- [x] Radio-specific BS parameters (LTE Band 3 / NR n78)
- [x] QNetwork architecture (14 → 256 → 128 → 64 → 4, GELU)
- [x] Top-k tower filtering and direction-aware scoring (helpers module)
- [x] DDQN handover decision logic in UserEquipment
- [x] Conditional Handover (CHO) post-processing — UE-side bearing/cosine similarity re-ranking
- [x] Trained DDQN agent evaluation vs 3GPP A3 baseline
- [x] Multi-seed A3 baseline testing (`test_rsrp.py`)
- [x] CHO weight sweep testing (`test_cho.py`)

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

## System-Level Abstraction Notes

This simulation is a **system-level abstraction** of the 5G NR physical layer, not a full PHY emulator. Key simplifying assumptions:

| Aspect | 3GPP Reality | Our Abstraction | Justification |
|---|---|---|---|
| **LOS/NLOS** | Probability function based on distance | Hard 5 m cutoff | Simulates dense urban canyon where UEs are almost always NLOS |
| **Antennas** | Massive MIMO with active beamforming | Static isotropic gain (`G_tx`) | Isolates handover algorithm evaluation from beam-tracking mechanics |
| **Path Loss** | Dual-slope model with breakpoint distances | Standard log-distance model | Widely accepted balance of realism and computational efficiency for RL |
| **RLF detection** | SINR/BLER-based Qout with T310/N310/N311 timers | Single RSRP threshold per-RAT (≈ -116 dBm) | Captures cell-edge failure events without full L1/RRC timer state machine |
| **Handover delay** | Variable interruption (search, SI acquisition, RACH) | Fixed per-RAT (NR: 20 ms, LTE: 40 ms) | Matches 3GPP intra-freq known-cell baseline from TS 38.133/36.133 |

See [architecture.md](architecture.md) for system architecture diagrams and [sources.md](sources.md) for full technical references and justifications.

---

## References

- 3GPP TR 38.901 — Channel model for frequencies from 0.5 to 100 GHz (shadow/fast fading parameters)
- 3GPP TS 36.104 — LTE Base Station radio transmission and reception (LTE power classes)
- 3GPP TS 38.104 — NR Base Station radio transmission and reception (NR power classes)
- 3GPP TS 38.133 — NR requirements for support of radio resource management (RLM Qout: Section 8.1.1, handover interruption: Section 8.2.2)
- 3GPP TS 36.133 — LTE requirements for support of radio resource management (RLM Qout: Section 7.6, handover interruption: Section 8.1.1.1)
- 3GPP TS 36.214 — LTE physical layer measurements (RSRP, RSRQ definitions)
- Gudmundson, M. — *Correlation Model for Shadow Fading in Mobile Radio Systems* (1991)
- Rappaport, T.S. — *Wireless Communications: Principles and Practice*

---

## License

MIT License — free to use, modify, and distribute.
