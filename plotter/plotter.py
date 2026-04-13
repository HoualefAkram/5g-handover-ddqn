"""
Data extraction and visualization for 5G Handover DDQN simulation results.

Reads TensorBoard event files from outputs/runs/, produces CSVs in plotter/csv/,
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
# Global style — bold, visible text on every graph
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "figure.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 10,
    }
)

# Canonical annotation font size (used for ax.annotate calls across all plots)
ANNOTATION_FONTSIZE = 9

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "outputs" / "runs"
CSV_DIR = BASE_DIR / "plotter" / "csv"

ALGORITHMS = ["A3_RSRP", "DDQN", "DDQN_CHO"]
SEED_COUNT = 10


def _detect_timestamp() -> str:
    """Auto-detect the most recent PERF timestamp from runs directory."""
    import re

    pattern = re.compile(r"^PERF_\w+_LONDON_(\d{8}_\d{6})$")
    timestamps = set()
    if RUNS_DIR.is_dir():
        for d in RUNS_DIR.iterdir():
            m = pattern.match(d.name)
            if m:
                timestamps.add(m.group(1))
    if not timestamps:
        raise FileNotFoundError(f"No PERF_*_LONDON_* runs found in {RUNS_DIR}")
    return max(timestamps)  # most recent


def _detect_training_run() -> str | None:
    """Auto-detect the most recent Training run, or None if absent."""
    import re

    pattern = re.compile(r"^Training_(\d{8}_\d{6})$")
    matches = []
    if RUNS_DIR.is_dir():
        for d in RUNS_DIR.iterdir():
            m = pattern.match(d.name)
            if m:
                matches.append(d.name)
    return max(matches) if matches else None


TIMESTAMP = _detect_timestamp()
TRAINING_RUN = _detect_training_run()

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


# --- 1b. Performance Metrics CSV (from PERF running-average files) ---
def extract_performance_csv() -> Path:
    rows = []  # (algorithm, seed_step, avg_handovers, avg_pingpongs, pingpong_rate)
    for algo in ALGORITHMS:
        ea = load_ea(f"PERF_{algo}_LONDON_{TIMESTAMP}")
        ho_events = ea.Scalars("Performance/AVERAGE_HANDOVERS")
        pp_events = ea.Scalars("Performance/AVERAGE_PINGPONG")
        pr_events = ea.Scalars("Performance/PINGPONG_RATE")
        ho_by_step = {e.step: e.value for e in ho_events}
        pp_by_step = {e.step: e.value for e in pp_events}
        pr_by_step = {e.step: e.value for e in pr_events}

        for step in sorted(ho_by_step):
            rows.append(
                (algo, step, ho_by_step[step], pp_by_step[step], pr_by_step[step])
            )

        # Final step (10) is already sum/10; multiply back for SUM row
        avg_ho = ho_by_step[SEED_COUNT]
        avg_pp = pp_by_step[SEED_COUNT]
        avg_pr = pr_by_step[SEED_COUNT]
        rows.append((algo, "SUM", avg_ho * SEED_COUNT, avg_pp * SEED_COUNT, avg_pr))
        rows.append((algo, "AVG", avg_ho, avg_pp, avg_pr))

    path = CSV_DIR / "performance_metrics.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "seed", "handovers", "pingpongs", "pingpong_rate"])
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
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color=color_r)
    ax1.plot(episodes, rewards, color=color_r, alpha=0.8, linewidth=0.8, label="Reward")
    ax1.tick_params(axis="y", labelcolor=color_r)

    # Axis 2: Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Loss", color=color_l)
    ax2.plot(episodes, losses, color=color_l, alpha=0.8, linewidth=0.8, label="Loss")
    ax2.tick_params(axis="y", labelcolor=color_l)

    # Axis 3: Epsilon
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.set_ylabel("Epsilon", color=color_e)
    ax3.plot(
        episodes, epsilons, color=color_e, alpha=0.8, linewidth=0.8, label="Epsilon"
    )
    ax3.tick_params(axis="y", labelcolor=color_e)

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("DDQN Training Metrics")
    fig.tight_layout()
    out = CSV_DIR.parent / "reward_loss.png"
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
    bars1 = ax.bar(
        x - width / 2,
        ho_vals,
        width,
        label="Handovers",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        pp_vals,
        width,
        label="Pingpongs",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.75,
    )

    ax.set_ylabel("Count")
    ax.set_title("Average Handovers & Pingpongs (10 Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Value annotations
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

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

    ax.set_xlabel("Averaged RSRP (Normalized)")
    ax.set_ylabel("Density")
    ax.set_title("RSRP Distribution (KDE) Across Algorithms")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
    bars1 = ax.bar(
        x - width / 2,
        ho_vals,
        width,
        label="Handovers",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        pp_vals,
        width,
        label="Pingpongs",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.75,
    )

    ax.set_ylabel("Count")
    ax.set_title("Total Handovers & Pingpongs (Sum of 10 Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    out = CSV_DIR.parent / "performance_bars_sum.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def _plot_ho_pprate_bars(csv_path: Path, agg_key: str, title: str, out_name: str):
    """Dual-axis grouped bar chart: Handovers (left) and Pingpong Rate % (right)."""
    data_ho = {}
    data_pr = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["seed"] == agg_key:
                algo = row["algorithm"]
                data_ho[algo] = float(row["handovers"])
                data_pr[algo] = float(row["pingpong_rate"]) * 100  # to %

    x_labels = [ALGO_DISPLAY[a] for a in ALGORITHMS]
    ho_vals = [data_ho[a] for a in ALGORITHMS]
    pr_vals = [data_pr[a] for a in ALGORITHMS]
    colors = [ALGO_COLORS[a] for a in ALGORITHMS]

    x = np.arange(len(ALGORITHMS))
    width = 0.32

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bars1 = ax1.bar(
        x - width / 2,
        ho_vals,
        width,
        label="Handovers",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Handovers")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        pr_vals,
        width,
        label="Pingpong Rate",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.75,
    )
    ax2.set_ylabel("Pingpong Rate (%)")

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    fig.suptitle(title)
    fig.tight_layout()
    out = CSV_DIR.parent / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_performance_pprate_avg(csv_path: Path):
    _plot_ho_pprate_bars(
        csv_path,
        "AVG",
        "Average Handovers & Pingpong Rate (10 Seeds)",
        "performance_pprate_avg.png",
    )


def plot_performance_pprate_sum(csv_path: Path):
    _plot_ho_pprate_bars(
        csv_path,
        "SUM",
        "Total Handovers & Pingpong Rate (Sum of 10 Seeds)",
        "performance_pprate_sum.png",
    )


def plot_reduction_vs_a3(csv_path: Path):
    """Bar chart showing DDQN / DDQN-CHO percentage reduction vs A3 baseline."""
    data_ho = {}
    data_pr = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["seed"] == "AVG":
                algo = row["algorithm"]
                data_ho[algo] = float(row["handovers"])
                data_pr[algo] = float(row["pingpong_rate"]) * 100  # to %

    base_ho = data_ho["A3_RSRP"]
    base_pr = data_pr["A3_RSRP"]

    algos = ["DDQN", "DDQN_CHO"]
    ho_reductions = [(base_ho - data_ho[a]) / base_ho * 100 for a in algos]
    pr_reductions = [(base_pr - data_pr[a]) / base_pr * 100 for a in algos]

    x_labels = [ALGO_DISPLAY[a] for a in algos]

    x = np.arange(len(algos))
    width = 0.32
    color_ho = "#3498db"
    color_pr = "#e67e22"

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(
        x - width / 2,
        ho_reductions,
        width,
        label="Handover Reduction",
        color=color_ho,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        pr_reductions,
        width,
        label="Pingpong Rate Reduction",
        color=color_pr,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.75,
    )

    ax.set_ylabel("Reduction vs A3-RSRP (%)")
    ax.set_title("Improvement Over A3-RSRP Baseline (10 Seeds Avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.59, 1.0))

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "reduction_vs_a3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_boxplot(csv_path: Path):
    """Box plot of RSRP distribution for each algorithm."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    fig, ax = plt.subplots(figsize=(8, 5))

    box_data = [algo_rsrp[a] for a in ALGORITHMS]
    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        labels=[ALGO_DISPLAY[a] for a in ALGORITHMS],
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    for patch, algo in zip(bp["boxes"], ALGORITHMS):
        patch.set_facecolor(ALGO_COLORS[algo])
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel("Averaged RSRP (Normalized)")
    ax.set_title("RSRP Distribution (Box Plot) Across Algorithms")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_boxplot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_violin(csv_path: Path):
    """Violin plot of RSRP distribution for each algorithm."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    fig, ax = plt.subplots(figsize=(8, 5))

    violin_data = [algo_rsrp[a] for a in ALGORITHMS]
    parts = ax.violinplot(
        violin_data,
        positions=range(len(ALGORITHMS)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    for body, algo in zip(parts["bodies"], ALGORITHMS):
        body.set_facecolor(ALGO_COLORS[algo])
        body.set_alpha(0.7)
        body.set_edgecolor("black")

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    parts["cmeans"].set_color("#e74c3c")
    parts["cmeans"].set_linewidth(1.2)
    parts["cmeans"].set_linestyle("--")
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)

    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGO_DISPLAY[a] for a in ALGORITHMS])
    ax.set_ylabel("Averaged RSRP (Normalized)")
    ax.set_title("RSRP Distribution (Violin Plot) Across Algorithms")
    ax.legend([parts["cmedians"], parts["cmeans"]], ["Median", "Mean"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_violin.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_fft(csv_path: Path):
    """FFT magnitude spectrum of RSRP signals for each algorithm."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    fig, ax = plt.subplots(figsize=(12, 5))

    for algo in ALGORITHMS:
        signal = np.array(algo_rsrp[algo])
        signal = signal - signal.mean()  # remove DC component
        n = len(signal)
        fft_mag = np.abs(np.fft.rfft(signal)) / n
        freqs = np.fft.rfftfreq(n)
        # skip DC bin (index 0)
        ax.plot(
            freqs[1:],
            fft_mag[1:],
            color=ALGO_COLORS[algo],
            alpha=0.8,
            linewidth=1.2,
            label=ALGO_DISPLAY[algo],
        )

    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT of RSRP Signal Across Algorithms")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_fft.png"
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


def plot_rsrp_mean_bar(csv_path: Path):
    """Bar chart of mean RSRP for each algorithm."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    x_labels = [ALGO_DISPLAY[a] for a in ALGORITHMS]
    means = [np.mean(algo_rsrp[a]) for a in ALGORITHMS]
    colors = [ALGO_COLORS[a] for a in ALGORITHMS]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        x_labels,
        means,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

    ax.set_ylabel("Mean RSRP (Normalized)")
    ax.set_title("Mean RSRP Across Algorithms (10 Seeds Avg)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_mean_bar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_std_bar(csv_path: Path):
    """Bar chart of RSRP standard deviation for each algorithm."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    x_labels = [ALGO_DISPLAY[a] for a in ALGORITHMS]
    stds = [np.std(algo_rsrp[a]) for a in ALGORITHMS]
    colors = [ALGO_COLORS[a] for a in ALGORITHMS]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        x_labels,
        stds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )

    ax.set_ylabel("RSRP Std Dev (Normalized)")
    ax.set_title("RSRP Standard Deviation Across Algorithms (10 Seeds Avg)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_std_bar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_cloud(csv_path: Path):
    """Cloud (strip) plot of RSRP values for each algorithm."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, algo in enumerate(ALGORITHMS):
        vals = np.array(algo_rsrp[algo])
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            i + jitter,
            vals,
            color=ALGO_COLORS[algo],
            alpha=0.08,
            s=6,
            edgecolors="none",
        )
        # Overlay mean marker
        ax.scatter(
            i,
            np.mean(vals),
            color=ALGO_COLORS[algo],
            s=120,
            marker="D",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"{ALGO_DISPLAY[algo]} (mean={np.mean(vals):.4f})",
        )

    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGO_DISPLAY[a] for a in ALGORITHMS])
    ax.set_ylabel("RSRP (Normalized)")
    ax.set_title("RSRP Cloud Plot Across Algorithms (10 Seeds Avg)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_cloud.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_raincloud(csv_path: Path):
    """Raincloud plot: half-violin + jittered strip + boxplot for RSRP."""
    algo_rsrp = {a: [] for a in ALGORITHMS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for algo in ALGORITHMS:
                val = row[algo]
                if val:
                    algo_rsrp[algo].append(float(val))

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)

    for i, algo in enumerate(ALGORITHMS):
        vals = np.array(algo_rsrp[algo])
        color = ALGO_COLORS[algo]

        # Half-violin (KDE) on top side
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(vals, bw_method=0.3)
        y_range = np.linspace(vals.min(), vals.max(), 300)
        density = kde(y_range)
        density = density / density.max() * 0.35  # scale width
        ax.fill_betweenx(
            y_range,
            i - density,
            i,
            color=color,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
        )

        # Boxplot on center
        bp = ax.boxplot(
            vals,
            positions=[i],
            widths=0.08,
            vert=True,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.9, edgecolor="black"),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
        )

        # Jittered strip on bottom side
        jitter = rng.uniform(0.05, 0.3, size=len(vals))
        ax.scatter(
            i + jitter,
            vals,
            color=color,
            alpha=0.06,
            s=4,
            edgecolors="none",
        )

    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([ALGO_DISPLAY[a] for a in ALGORITHMS])
    ax.set_ylabel("RSRP (Normalized)")
    ax.set_title("RSRP Raincloud Plot Across Algorithms (10 Seeds Avg)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_raincloud.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_raw(csv_path: Path):
    """Raw RSRP values over simulation timesteps for each algorithm."""
    data = _read_rsrp_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    for algo in ALGORITHMS:
        steps, vals = data[algo]
        ax.plot(
            steps,
            vals,
            color=ALGO_COLORS[algo],
            alpha=0.6,
            linewidth=0.5,
            label=ALGO_DISPLAY[algo],
        )

    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Averaged RSRP (Normalized)")
    ax.set_title("Raw RSRP Over Time (Averaged Across 10 Seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
        ax.plot(
            steps,
            ema,
            color=ALGO_COLORS[algo],
            linewidth=1.8,
            label=f"{ALGO_DISPLAY[algo]} (EMA {span})",
        )

    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("RSRP (Normalized)")
    ax.set_title("RSRP with Exponential Moving Average")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_ema.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


def plot_rsrp_ema_zoomed(csv_path: Path, span: int = 100):
    """RSRP EMA with y-axis zoomed to data range so differences are visible."""
    data = _read_rsrp_csv(csv_path)
    alpha = 2.0 / (span + 1)

    ema_data = {}
    global_min, global_max = 1.0, 0.0

    for algo in ALGORITHMS:
        steps, vals = data[algo]
        vals_arr = np.array(vals)
        ema = np.empty_like(vals_arr)
        ema[0] = vals_arr[0]
        for i in range(1, len(vals_arr)):
            ema[i] = alpha * vals_arr[i] + (1 - alpha) * ema[i - 1]
        ema_data[algo] = (np.array(steps), ema)
        global_min = min(global_min, ema.min())
        global_max = max(global_max, ema.max())

    # Add padding and convert to percentage
    pad = (global_max - global_min) * 0.15
    y_lo = max(0, global_min - pad)
    y_hi = min(1, global_max + pad)

    fig, ax = plt.subplots(figsize=(12, 6))

    for algo in ALGORITHMS:
        steps, ema = ema_data[algo]
        ax.plot(
            steps,
            ema * 100,
            color=ALGO_COLORS[algo],
            linewidth=2,
            label=ALGO_DISPLAY[algo],
        )

    ax.set_ylim(y_lo * 100, y_hi * 100)
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("RSRP Signal Quality (%)")
    ax.set_title(f"RSRP — Exponential Moving Average (span={span}, zoomed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = CSV_DIR.parent / "rsrp_ema_zoomed.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {out.name}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"Detected PERF timestamp: {TIMESTAMP}")
    print(f"Detected Training run:   {TRAINING_RUN or '(none)'}")

    print("\nPhase 1: Extracting CSVs ...")
    training_csv = None
    if TRAINING_RUN:
        training_csv = extract_training_csv()
    else:
        print("  [SKIP] No Training run found — skipping training CSV")
    perf_csv = extract_performance_csv()
    rsrp_csv = extract_rsrp_csv()

    print("\nPhase 2: Generating plots ...")
    if training_csv:
        plot_training(training_csv)
    else:
        print("  [SKIP] No training data — skipping training plot")
    plot_performance_bars(perf_csv)
    plot_performance_bars_sum(perf_csv)
    plot_rsrp_kde(rsrp_csv)
    plot_rsrp_raw(rsrp_csv)
    plot_performance_pprate_avg(perf_csv)
    plot_performance_pprate_sum(perf_csv)
    plot_rsrp_ema(rsrp_csv)
    plot_rsrp_ema_zoomed(rsrp_csv)
    plot_reduction_vs_a3(perf_csv)
    plot_rsrp_boxplot(rsrp_csv)
    plot_rsrp_violin(rsrp_csv)
    plot_rsrp_fft(rsrp_csv)
    plot_rsrp_mean_bar(rsrp_csv)
    plot_rsrp_std_bar(rsrp_csv)
    plot_rsrp_cloud(rsrp_csv)
    plot_rsrp_raincloud(rsrp_csv)

    print("\nDone.")
