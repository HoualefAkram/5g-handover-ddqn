# Architecture — 5G Handover DDQN

## High-Level Pipeline

```mermaid
graph LR
    subgraph Prepare["prepare.py"]
        P1["Download OSM Map"]
        P2["Download Towers"]
        P3["Generate SUMO Traffic"]
    end

    subgraph Train["rl/ddqn_agent.py"]
        T1["HandoverEnv\n(Gymnasium)"]
        T2["DDQN Training Loop"]
        T3["Export Model"]
    end

    subgraph Evaluate["test.py / test_rsrp.py"]
        E1["A3_RSRP_3GPP"]
        E2["DDQN"]
        E3["DDQN_CHO"]
    end

    subgraph Output["Outputs"]
        O1["TensorBoard Logs"]
        O2["Folium Map"]
        O3["Plotter Charts"]
    end

    Prepare --> Train
    Prepare --> Evaluate
    T3 -->|"final_ddqn_model.pth"| E2
    T3 -->|"final_ddqn_model.pth"| E3
    Evaluate --> Output
    Train -->|"TensorBoard"| O1

    classDef prep fill:#50b86c,stroke:#2d7a42,color:#fff
    classDef train fill:#9b59b6,stroke:#6c3483,color:#fff
    classDef eval fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef out fill:#e8a838,stroke:#b07c20,color:#fff

    class P1,P2,P3 prep
    class T1,T2,T3 train
    class E1,E2,E3 eval
    class O1,O2,O3 out
```

## Data Preparation — prepare.py

```mermaid
graph TB
    PARAMS["Params\nMAP_TOP_LEFT · MAP_BOTTOM_RIGHT\nMCC=234 · SEED=100\nSIMULATION_TIME=300 · STEP_LENGTH=0.1"]

    subgraph OSM["Map Download"]
        OVERPASS["Overpass API\n(OpenStreetMap)"]
        OSM_CACHE["cache/maps/map.osm\n(skipped if bbox matches)"]
        OVERPASS --> OSM_CACHE
    end

    subgraph TOWER["Tower Download"]
        OPENCELLID["OpenCellID API\n(.env API key)"]
        TOWER_CSV["Download CSV by MCC"]
        TOWER_FILTER["Filter LTE/NR to bbox\nParse cell_id → tower_id\nDeduplicate"]
        TOWER_CACHE["cache/towers/towers.json"]
        OPENCELLID --> TOWER_CSV --> TOWER_FILTER --> TOWER_CACHE
    end

    subgraph SUMO["SUMO Traffic Generation"]
        NETCONVERT["netconvert\nOSM → .net.xml"]
        RANDOM_TRIPS["randomTrips.py\n(seed-based)"]
        DUAROUTER["duarouter\nTrips → Routes"]
        SUMO_SIM["sumo\nSimulate 300s @ 0.1s steps"]
        FCD_OUT["outputs/sumo/trace.xml\n(FCD trace)"]
        NETCONVERT --> RANDOM_TRIPS --> DUAROUTER --> SUMO_SIM --> FCD_OUT
    end

    PARAMS --> OSM
    PARAMS --> TOWER
    PARAMS --> SUMO
```

## Data Models

```mermaid
graph TB
    subgraph BS_MODEL["BaseTower"]
        BS_LTE["BaseTower.LTE()\nBand 3 · 1800 MHz\nP_tx=46 dBm · G_tx=15 dBi\nBW=20 MHz"]
        BS_NR["BaseTower.NR()\nn78 · 3500 MHz\nP_tx=43 dBm · G_tx=17 dBi\nBW=100 MHz"]
    end

    subgraph UE_MODEL["UserEquipment"]
        UE_PROPS["id · latlng · speed · angle\ng_rx=0 dBi · serving_bs\npath_history · connection_history\ngenerated_reports · rlf_count · dho_time"]
        UE_MOVE["move_to(latlng, timestep, speed, angle)"]
        UE_REPORT["generate_report(all_bs, timestep)\n→ NGRANReport"]
        UE_HO_CHECK["Handover Check\n(algorithm-dependent)"]
        UE_CONNECT["connect_to_tower() / handover()"]
        UE_METRICS["get_total_handovers()\nget_total_pingpong()\nget_pingpong_rate()\nget_normalized_time_since_last_handover()"]

        UE_MOVE --> UE_REPORT --> UE_HO_CHECK --> UE_CONNECT
    end

    subgraph REPORT_MODEL["NGRANReport"]
        REPORT_FIELDS["ue_id · timestep\nrsrp_values: dict[bs_id → index]\nrsrq_values: dict[bs_id → index]"]
    end

    subgraph FCD_MODEL["CarFcdData"]
        FCD_FIELDS["car_id · latlng · angle\nspeed · timestep"]
    end
```

## Signal Model — WaveUtils

### RSRP Calculation Chain

```mermaid
graph LR
    DIST["Haversine\ndistance(UE, BS)"]
    LOS{"d ≤ 5m?"}
    N_LOS["n = 2.0"]
    N_NLOS["n = 3.5"]
    PL["Path Loss\nPL(d₀) + 10·n·log₁₀(d/d₀)"]
    SF["Shadow Fading\nGudmundson model\nσ_LOS=6.0 dB · σ_NLOS=8.93 dB\nd_corr=10 m"]
    FF["Fast Fading\nLOS: Rician (K=7 dB)\nNLOS: Rayleigh"]
    RSRP["RSRP (dBm)\nP_tx + G_tx + G_rx\n− PL − Shadow + Fast"]

    DIST --> LOS
    LOS -->|Yes| N_LOS --> PL
    LOS -->|No| N_NLOS --> PL
    DIST --> SF
    DIST --> FF
    PL --> RSRP
    SF --> RSRP
    FF --> RSRP

    style RSRP fill:#e8a838,color:#fff,font-weight:bold
```

### RSRQ and Index Mapping

```mermaid
graph LR
    RSRP_IN["RSRP (dBm)\nper BS"]
    NOISE["Thermal Noise\n−174 + 10·log₁₀(BW) + NF\nLTE: ≈−100 dBm · NR: ≈−87 dBm"]
    RSSI["RSSI = Σ 10^(RSRP_i/10)\n+ 10^(noise/10)"]
    RSRQ["RSRQ = RSRP − RSSI (dB)"]

    IDX_RSRP["RSRP Index\nLTE: 0–97 · floor(dBm + 141)\nNR: 0–127 · floor(dBm + 157)"]
    IDX_RSRQ["RSRQ Index\nLTE: 0–34 · floor((dB + 20) / 0.5)\nNR: 0–127 · floor((dB + 43.5) / 0.5)"]
    NORM["Normalize to [0,1]\nRSRP: idx/97 or idx/127"]

    RSRP_IN --> RSSI
    NOISE --> RSSI
    RSSI --> RSRQ
    RSRP_IN --> IDX_RSRP --> NORM
    RSRQ --> IDX_RSRQ

    style NORM fill:#e8a838,color:#fff,font-weight:bold
```

## Handover Algorithms

### A3_RSRP_3GPP

```mermaid
graph LR
    CHECK_SERVING{"Serving BS\nexists?"}
    ATTACH["Attach to strongest\navailable tower"]
    A3_COND["RSRP(neighbor) >\nRSRP(serving) + 2 dB?\n(compared in dBm)"]
    TTT["TTT Check: 160 ms\nCondition must hold for\nceil(0.16 / Δt) + 1 reports"]
    RESULT["Return target BS\n(or None)"]

    CHECK_SERVING -->|No| ATTACH
    CHECK_SERVING -->|Yes| A3_COND
    A3_COND -->|No| RESULT
    A3_COND -->|Yes| TTT --> RESULT

    classDef decision fill:#d94a6e,stroke:#a12d4d,color:#fff
    class A3_COND,TTT decision
```

### DDQN (Pure)

```mermaid
graph LR
    CHECK{"Serving BS\nexists?"}
    ATTACH["Attach to strongest"]
    TOP4["Top-4 Filter\nby normalized RSRP\n(serving guaranteed)"]
    STATE["Build state (14 features)\n4×RSRP + 4×RSRP_trend\n+ 4×one_hot + 1×speed\n+ 1×time_since_ho"]
    QNET["QNetwork forward\n14 → 256 → 128 → 64 → 4"]
    ARGMAX["argmax Q-values\n→ select tower"]
    DECIDE{"target ==\nserving?"}
    HO["Return target BS"]
    STAY["Return None\n(no handover)"]

    CHECK -->|No| ATTACH
    CHECK -->|Yes| TOP4 --> STATE --> QNET --> ARGMAX --> DECIDE
    DECIDE -->|Yes| STAY
    DECIDE -->|No| HO

    classDef rl fill:#9b59b6,stroke:#6c3483,color:#fff
    class QNET,ARGMAX rl
```

### DDQN_CHO (Confidence-Gated Conditional Handover)

```mermaid
graph TB
    CHECK_SERVING{"Serving BS\nexists?"}
    ATTACH["Attach to strongest"]
    TOP4["Top-4 Filter\nby normalized RSRP\n(serving guaranteed)"]
    STATE["Build state vector (14 features)\n4×RSRP + 4×RSRP_trend\n+ 4×serving_one_hot\n+ 1×speed + 1×time_since_ho"]
    QNET["QNetwork forward pass\n14 → 256 → 128 → 64 → 4"]
    STAY_CHECK{"DDQN pick ==\nserving?"}
    NO_HO["Return None\n(no handover)"]
    SOFTMAX["Softmax top-2 Q-values"]
    GATE{"softmax(Q_best)\n≥ confidence_threshold\n(default 0.55)?"}
    TRUST["Trust DDQN pick\n→ handover to argmax"]
    FALLBACK["Weighted Fallback\nq_weight = clamp(RSRP_gap, 0, 1)\nsim_weight = 1 − q_weight"]
    SCORE["score = sim_weight × cos_similarity\n+ q_weight × softmax_Q\nfor each top-2 candidate"]
    BEST["Return candidate\nwith higher score"]

    CHECK_SERVING -->|No| ATTACH
    CHECK_SERVING -->|Yes| TOP4 --> STATE --> QNET --> STAY_CHECK
    STAY_CHECK -->|Yes| NO_HO
    STAY_CHECK -->|No| SOFTMAX --> GATE
    GATE -->|Yes| TRUST
    GATE -->|No| FALLBACK --> SCORE --> BEST

    classDef decision fill:#d94a6e,stroke:#a12d4d,color:#fff
    classDef rl fill:#9b59b6,stroke:#6c3483,color:#fff
    class GATE,STAY_CHECK decision
    class QNET,SOFTMAX,SCORE rl
```

## QNetwork Architecture

```mermaid
graph LR
    INPUT["Input (14)\n4×RSRP\n4×RSRP trend\n4×serving one-hot\n1×speed\n1×time since HO"]
    L1["Linear(14→256)\nGELU"]
    L2["Linear(256→128)\nGELU"]
    L3["Linear(128→64)\nGELU"]
    L4["Linear(64→4)"]
    OUTPUT["Q-values\n(1 per top-4 tower)"]

    INPUT --> L1 --> L2 --> L3 --> L4 --> OUTPUT

    style INPUT fill:#5dade2,color:#fff
    style OUTPUT fill:#e8a838,color:#fff
    style L1 fill:#9b59b6,color:#fff
    style L2 fill:#9b59b6,color:#fff
    style L3 fill:#9b59b6,color:#fff
    style L4 fill:#9b59b6,color:#fff
```

## RL Training — ddqn_agent.py

### Environment and Reward

```mermaid
graph TB
    subgraph ENV["HandoverEnv (Gymnasium)"]
        SPACES["Action: Discrete(4)\nObs: Box(−1, 1, shape=(14,))"]
        RESET["reset()\n→ new SUMO traffic (random seed)\n→ reset fading state\n→ agent = UE 0 (NONE algorithm)\n→ retries if < 10 agent steps"]
        STEP["step(action)\n1. Compute dynamic handover penalty\n2. Early terminate if car reached destination\n3. Execute handover or initial connection\n4. Move agent (generates report internally)\n5. Update top-4 towers\n6. Calculate reward + RLF penalty"]
    end

    subgraph REWARD["Reward Design"]
        RW_HO["Handover executed:\nrsrp(new) − rsrp(old)\n− cooldown_penalty × (1 − time_since_ho)\n(both measured at same position after move)"]
        RW_STAY["Stay (no handover):\nrsrp(serving) − rsrp(best_alternative)\n(counterfactual comparison)"]
        RW_RLF["RLF penalty: −1.5\nwhen serving RSRP < threshold\n(NR: 41/127 · LTE: 25/97)"]
    end

    STEP --> REWARD
```

### Training Loop

```mermaid
graph TB
    EP_START["state, _ = env.reset()"]
    EP_ACTION["ε-greedy action\nrandom if rand < ε\nelse argmax Q_policy(s)"]
    EP_STEP["new_state, reward, done\n= env.step(action)"]
    EP_STORE["memory.append(\nstate, action, reward,\nnew_state, done)"]
    EP_TRAIN{"len(memory) ≥ 1000\nAND step % 20 == 0?"}
    EP_SAMPLE["Sample batch of 128"]
    EP_BELLMAN["DDQN Target:\na* = argmax Q_policy(s')\nV = Q_target(s', a*)\ny = r + 0.97·V·(1−done)"]
    EP_LOSS["Huber Loss:\nL = SmoothL1(\nQ_policy(s,a), y)"]
    EP_BACKPROP["adam.zero_grad()\nloss.backward()\nadam.step()"]
    EP_DONE{"terminated\nor truncated?"}
    EP_TARGET{"(episode+1)\n% 2 == 0?"}
    EP_HARD["target.hard_update(policy)"]
    EP_DECAY["ε = max(0.05, ε × 0.99)"]
    EP_SAVE["Save checkpoint + replay buffer"]
    EP_LOG["TensorBoard Log:\nreward · loss · Q · ε\nhandovers · ping-pongs\nRSRP · RSRQ"]

    EP_START --> EP_ACTION --> EP_STEP --> EP_STORE --> EP_TRAIN
    EP_TRAIN -->|No| EP_DONE
    EP_TRAIN -->|Yes| EP_SAMPLE --> EP_BELLMAN --> EP_LOSS --> EP_BACKPROP --> EP_DONE
    EP_DONE -->|No| EP_ACTION
    EP_DONE -->|Yes| EP_TARGET
    EP_TARGET -->|Yes| EP_HARD --> EP_DECAY
    EP_TARGET -->|No| EP_DECAY
    EP_DECAY --> EP_SAVE --> EP_LOG

    classDef rl fill:#9b59b6,stroke:#6c3483,color:#fff
    classDef log fill:#5dade2,stroke:#2e86c1,color:#fff
    class EP_BELLMAN,EP_LOSS,EP_BACKPROP rl
    class EP_LOG log
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 800 |
| Learning rate | 5e-4 (Adam) |
| Discount (γ) | 0.97 |
| ε start | 1.0 |
| ε decay | 0.99 per episode |
| ε min | 0.05 |
| Batch size | 128 |
| Min buffer | 1000 |
| Max buffer | 50000 |
| Train every | 20 steps |
| Target update | every 2 episodes (hard copy) |
| Loss | SmoothL1 (Huber) |

## Ping-Pong Detection

```mermaid
graph LR
    subgraph HISTORY["connection_history"]
        H1["(BS_A, t₁)"]
        H2["(BS_B, t₂)"]
        H3["(BS_A, t₃)"]
    end

    H1 --> H2 --> H3

    CHECK{"BS_A == BS_A\nAND BS_A ≠ BS_B\nAND (t₃ − t₂) < 1.0s"}
    H3 --> CHECK
    CHECK -->|Yes| PP["Ping-Pong +1"]
    CHECK -->|No| SKIP["Not a ping-pong"]
```

## Simulation — test.py

```mermaid
graph TB
    LOAD["Load towers from cache\nParse FCD trace"]

    subgraph A3_RUN["A3 RSRP Run"]
        A3_INIT["Create all UEs\nalgorithm = A3_RSRP_3GPP"]
        A3_SIM["For each timestep:\n→ move cars (FCD data)\n→ generate reports\n→ check handover\n→ log per-UE + system metrics"]
        A3_INIT --> A3_SIM
    end

    RESET["Reset fading state\nClear BS connections"]

    subgraph DDQN_RUN["DDQN-CHO Run"]
        DDQN_LOAD["Load trained model\n(CUDA if available)"]
        DDQN_INIT["Create all UEs\nalgorithm = DDQN_CHO"]
        DDQN_SIM["Same simulation loop\nwith DDQN-CHO decisions"]
        DDQN_LOAD --> DDQN_INIT --> DDQN_SIM
    end

    subgraph OUT["Outputs"]
        FOLIUM["Folium Map\nBS markers (black=LTE, red=NR)\nUE paths + start/end markers"]
        TB["TensorBoard\noutputs/runs/*"]
    end

    LOAD --> A3_RUN --> RESET --> DDQN_RUN --> OUT

    classDef eval fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef out fill:#e8a838,stroke:#b07c20,color:#fff
    class A3_INIT,A3_SIM,DDQN_LOAD,DDQN_INIT,DDQN_SIM eval
    class FOLIUM,TB out
```

## TensorBoard Logging

```mermaid
graph LR
    subgraph UE_METRICS["Per-UE Metrics (UE_{id}/)"]
        M1["RSRP · RSRQ\nTOTAL_HANDOVERS\nTOTAL_PINGPONG\nPINGPONG_RATE\nTOTAL_RLF · TOTAL_DHO"]
    end

    subgraph GLOBAL_METRICS["System Metrics (Performance/)"]
        M2["AVERAGE_RSRP · AVERAGE_RSRQ\nTOTAL/AVERAGE_HANDOVERS\nTOTAL/AVERAGE_PINGPONG\nPINGPONG_RATE · AVERAGE_PINGPONG_RATE\nTOTAL/AVERAGE_RLF\nTOTAL/AVERAGE_DHO"]
    end

    subgraph TRAINING_METRICS["Training Metrics (Performance/)"]
        M3["Total_Reward · Episode_Length\nEpsilon · Average_Loss\nAverage_Max_Q"]
    end

    SIM["test.py / test_rsrp.py"] --> UE_METRICS
    SIM --> GLOBAL_METRICS
    TRAIN["rl/ddqn_agent.py"] --> TRAINING_METRICS
    TRAIN --> UE_METRICS
```
