"""
Data extraction and visualization for 5G Handover DDQN simulation results.

Reads TensorBoard event files from outputs/runs/, produces CSVs in ignored/csv/,
and generates Matplotlib/Seaborn visualizations.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "outputs" / "runs"
CSV_DIR = BASE_DIR / "ignored" / "csv"

TRAINING_RUN = "Training_20260405_134307"
TIMESTAMP = "20260407_151337"
ALGORITHMS = ["A3_RSRP", "DDQN", "DDQN_CHO"]
SEED_COUNT = 10

ALGO_DISPLAY = {
    "A3_RSRP": "A3-RSRP (3GPP)",
    "DDQN": "DDQN",
    "DDQN_CHO": "DDQN-CHO",
}
ALGO_COLORS = {
    "A3_RSRP": "#e74c3c",
    "DDQN": "#3498db",
    "DDQN_CHO": "#2ecc71",
}

CSV_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_ea(run_name: str) -> EventAccumulator:
    ea = EventAccumulator(str(RUNS_DIR / run_name))
    ea.Reload()
    return ea


# ===================================================================
# Phase 1 - CSV Extraction
# ===================================================================

# --- 1a. Training Metrics CSV ---
def extract_training_csv() -> Path:
    ea = load_ea(TRAINING_RUN)
    rewards = {e.step: e.value for e in ea.Scalars("Performance/Total_Reward")}
    losses = {e.step: e.value for e in ea.Scalars("Performance/Average_Loss")}
    epsilons = {e.step: e.value for e in ea.Scalars("Performance/Epsilon")}

    episodes = sorted(rewards.keys())
    path = CSV_DIR / "training_metrics.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "loss", "epsilon"])
        for ep in episodes:
            w.writerow([ep, rewards[ep], losses.get(ep, ""), epsilons.get(ep, "")])
    print(f"  [CSV] {path.name}: {len(episodes)} episodes")
    return path


# --- 1b. Performance Metrics CSV (per-seed from SEED files) ---
def extract_performance_csv() -> Path:
    rows = []  # (algorithm, seed, handovers, pingpongs)
    for algo in ALGORITHMS:
        seed_handovers = []
        seed_pingpongs = []
        for seed_idx in range(1, SEED_COUNT + 1):
            run = f"SEED{str(seed_idx).zfill(2)}_{algo}_LONDON_{TIMESTAMP}"
            ea = load_ea(run)
            ho = ea.Scalars("UE_0/TOTAL_HANDOVERS")[-1].value
            pp = ea.Scalars("UE_0/TOTAL_PINGPONG")[-1].value
            rows.append((algo, seed_idx, ho, pp))
            seed_handovers.append(ho)
            seed_pingpongs.append(pp)
        # Aggregate rows: sum and average
        rows.append((algo, "SUM", sum(seed_handovers), sum(seed_pingpongs)))
        rows.append((
            algo,
            "AVG",
            sum(seed_handovers) / SEED_COUNT,
            sum(seed_pingpongs) / SEED_COUNT,
        ))

    path = CSV_DIR / "performance_metrics.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "seed", "handovers", "pingpongs"])
        for r in rows:
            w.writerow(r)
    print(f"  [CSV] {path.name}: {len(rows)} rows")
    return path


# --- 1c. RSRP Distribution CSV (averaged per-timestep from PERF files) ---
def extract_rsrp_csv() -> Path:
    all_data = {}
    max_steps = 0
    for algo in ALGORITHMS:
        ea = load_ea(f"PERF_{algo}_LONDON_{TIMESTAMP}")
        events = ea.Scalars("Performance/AVERAGE_RSRP")
        all_data[algo] = {e.step: e.value for e in events}
        max_steps = max(max_steps, max(e.step for e in events))

    path = CSV_DIR / "rsrp_distribution.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + ALGORITHMS)
        for step in range(max_steps + 1):
            row = [step]
            for algo in ALGORITHMS:
                row.append(all_data[algo].get(step, ""))
            w.writerow(row)
    print(f"  [CSV] {path.name}: {max_steps + 1} timesteps")
    return path


# ===================================================================
# Phase 2 - Visualizations
# ===================================================================

def plot_training(csv_path: Path):
    """Multi-axis training plot: Reward, Loss, Epsilon."""
    episodes, rewards, losses, epsilons = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            losses.append(float(row["loss"]))
            epsilons.append(float(row["epsilon"]))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_r = "#2980b9"
    color_l = "#e74c3c"
    color_e = "#27ae60"

    # Axis 1: Reward
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Total Reward", color=color_r, fontsize=12)
    ax1.plot(episodes, rewards, color=color_r, alpha=0.8, linewidth=0.8, label="Reward")
    ax1.tick_params(axis="y", labelcolor=color_r)

    # Axis 2: Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Loss", color=color_l, fontsize=12)
    ax2.plot(episodes, losses, color=color_l, alpha=0.8, linewidth=0.8, label="Loss")
    ax2.tick_params(axis="y", labelcolor=color_l)

    # Axis 3: Epsilon
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.set_ylabel("Epsilon", color=color_e, fontsize=12)
    ax3.plot(episodes, epsilons, color=color_e, alpha=0.8, linewidth=0.8, label="Epsilon")
    ax3.tick_params(axis="y", labelcolor=color_e)

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    fig.suptitle("DDQN Training Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = CSV_DIR.parent / "training_plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_performance_bars(csv_path: Path):
    """Grouped bar chart: Handovers and Pingpongs per algorithm."""
    avg_ho = {}
    avg_pp = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["seed"] == "AVG":
                algo = row["algorithm"]
                avg_ho[algo] = float(row["handovers"])
                avg_pp[algo] = float(row["pingpongs"])

    x_labels = [ALGO_DISPLAY[a] for a in ALGORITHMS]
    ho_vals = [avg_ho[a] for a in ALGORITHMS]
    pp_vals = [avg_pp[a] for a in ALGORITHMS]
    colors = [ALGO_COLORS[a] for a in ALGORITHMS]

    x = np.arange(len(ALGORITHMS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, ho_vals, width, label="Handovers",
                   color=colors, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, pp_vals, width, label="Pingpongs",
                   color=colors, edgecolor="black", linewidth=0.5,
                   hatch="///", alpha=0.75)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Average Handovers & Pingpongs (10 Seeds)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.legend(fontsize=10)

    # Value annotations
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    out = CSV_DIR.parent / "performance_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_kde(csv_path: Path):
    """Overlay KDE curves for RSRP distribution of each algorithm."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in ALGORITHMS:
        sns.kdeplot(
            algo_rsrp[algo],
            label=ALGO_DISPLAY[algo],
            color=ALGO_COLORS[algo],
            linewidth=2,
            ax=ax,
        )

    ax.set_xlabel("Averaged RSRP (Normalized)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("RSRP Distribution (KDE) Across Algorithms", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_kde.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_performance_bars_sum(csv_path: Path):
    """Grouped bar chart: Total (sum) Handovers and Pingpongs per algorithm."""
    sum_ho = {}
    sum_pp = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["seed"] == "SUM":
                algo = row["algorithm"]
                sum_ho[algo] = float(row["handovers"])
                sum_pp[algo] = float(row["pingpongs"])

    x_labels = [ALGO_DISPLAY[a] for a in ALGORITHMS]
    ho_vals = [sum_ho[a] for a in ALGORITHMS]
    pp_vals = [sum_pp[a] for a in ALGORITHMS]
    colors = [ALGO_COLORS[a] for a in ALGORITHMS]

    x = np.arange(len(ALGORITHMS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, ho_vals, width, label="Handovers",
                   color=colors, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, pp_vals, width, label="Pingpongs",
                   color=colors, edgecolor="black", linewidth=0.5,
                   hatch="///", alpha=0.75)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Total Handovers & Pingpongs (Sum of 10 Seeds)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.legend(fontsize=10)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    out = CSV_DIR.parent / "performance_bars_sum.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def _read_rsrp_csv(csv_path: Path) -> dict[str, tuple[list, list]]:
    """Read RSRP CSV and return {algo: (steps, values)} with no gaps."""
    algo_steps = {a: [] for a in ALGORITHMS}
    algo_vals = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_steps[algo].append(int(row["step"]))
                    algo_vals[algo].append(float(val))
    return {a: (algo_steps[a], algo_vals[a]) for a in ALGORITHMS}


def plot_rsrp_raw(csv_path: Path):
    """Raw RSRP values over simulation timesteps for each algorithm."""
    data = _read_rsrp_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    for algo in ALGORITHMS:
        steps, vals = data[algo]
        ax.plot(steps, vals, color=ALGO_COLORS[algo], alpha=0.6,
                linewidth=0.5, label=ALGO_DISPLAY[algo])

    ax.set_xlabel("Simulation Timestep", fontsize=12)
    ax.set_ylabel("Averaged RSRP (Normalized)", fontsize=12)
    ax.set_title("Raw RSRP Over Time (Averaged Across 10 Seeds)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_raw.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_ema(csv_path: Path, span: int = 100):
    """RSRP with Exponential Moving Average smoothing."""
    data = _read_rsrp_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    alpha = 2.0 / (span + 1)

    for algo in ALGORITHMS:
        steps, vals = data[algo]
        vals_arr = np.array(vals)

        # Compute EMA
        ema = np.empty_like(vals_arr)
        ema[0] = vals_arr[0]
        for i in range(1, len(vals_arr)):
            ema[i] = alpha * vals_arr[i] + (1 - alpha) * ema[i - 1]

        ax.plot(steps, vals_arr, color=ALGO_COLORS[algo], alpha=0.15, linewidth=0.4)
        ax.plot(steps, ema, color=ALGO_COLORS[algo], linewidth=1.8,
                label=f"{ALGO_DISPLAY[algo]} (EMA {span})")

    ax.set_xlabel("Simulation Timestep", fontsize=12)
    ax.set_ylabel("RSRP (Normalized)", fontsize=12)
    ax.set_title("RSRP with Exponential Moving Average", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_ema.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Phase 1: Extracting CSVs ...")
    training_csv = extract_training_csv()
    perf_csv = extract_performance_csv()
    rsrp_csv = extract_rsrp_csv()

    print("\nPhase 2: Generating plots ...")
    plot_training(training_csv)
    plot_performance_bars(perf_csv)
    plot_performance_bars_sum(perf_csv)
    plot_rsrp_kde(rsrp_csv)
    plot_rsrp_raw(rsrp_csv)
    plot_rsrp_ema(rsrp_csv)

    print("\nDone.")
