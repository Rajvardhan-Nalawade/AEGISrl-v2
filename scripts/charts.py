import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colours: fixed per agent, never by rank. Learned agents use the first three
# categorical slots (validated colour-blind safe); rule baselines are neutral grey.
COLORS = {"tabular": "#2a78d6", "dqn": "#eb6834", "ppo": "#1baf7a",
          "rbac": "#9a9893", "risk_rules": "#6f6d68"}
LABELS = {"tabular": "Tabular Q", "dqn": "DQN", "ppo": "PPO",
          "rbac": "RBAC", "risk_rules": "Risk rules"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def chart_seed_returns(seeds: dict, out: Path):
    """Mean +/- std of overall return, with every individual seed visible."""
    algos = [a for a in ("tabular", "dqn", "ppo") if a in seeds]
    fig, ax = plt.subplots(figsize=(7, 2.6), facecolor=SURFACE)
    style(ax)
    for y, algo in enumerate(algos):
        values = np.array([r["overall"] for r in seeds[algo]])
        mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
        ax.scatter(values, np.full_like(values, y), s=36, color=COLORS[algo],
                   alpha=0.45, linewidths=0, zorder=2)
        ax.errorbar(mean, y, xerr=std, fmt="o", ms=9, color=COLORS[algo],
                    mew=0, elinewidth=2, capsize=0, zorder=3)
        ax.annotate(f"{mean:.2f} ± {std:.2f}", (mean, y), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(algos)), [LABELS[a] for a in algos], color=INK, fontsize=10)
    ax.set_ylim(-0.6, len(algos) - 0.3)
    ax.set_xlabel("Average return per session (higher is better)", color=INK_2, fontsize=9)
    ax.set_title(f"Return across {len(seeds[algos[0]])} training seeds",
                 loc="left", color=INK, fontsize=12, fontweight="bold")
    save(fig, out)


def chart_tradeoff(seeds: dict, out: Path):
    """Security vs friction: data leaked against how often real users are challenged."""
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for algo in ("tabular", "dqn", "ppo"):
        if algo not in seeds:
            continue
        x = np.array([r["legit_challenge_rate"] for r in seeds[algo]]) * 100
        y = np.array([r["leak_per_attack"] for r in seeds[algo]])
        ax.scatter(x, y, s=70, color=COLORS[algo], alpha=0.85, linewidths=0,
                   label=LABELS[algo], zorder=3)
        ax.annotate(LABELS[algo], (np.median(x), y.min()), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top", fontsize=10, color=INK)
    ax.set_xlabel("Legitimate requests challenged with MFA (%)  →  more friction",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("Sensitive data leaked per attack session\n↓ better", color=INK_2, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.set_title("Security vs. friction (one dot per training seed)",
                 loc="left", color=INK, fontsize=12, fontweight="bold")
    save(fig, out)


def chart_robustness(robust: dict, out: Path):
    """Share of attack sessions fully stopped, per scenario (small multiples)."""
    scenarios = list(robust)
    titles = {"standard": "Standard", "noisy_signals": "Noisy signals (×1.5)",
              "realistic_mix": "Realistic mix (95% legit)", "adaptive_wave": "Adaptive-attacker wave"}
    order = [a for a in ("rbac", "risk_rules", "tabular", "dqn", "ppo") if a in robust[scenarios[0]]]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(12, 3.2), sharey=True, facecolor=SURFACE)
    for ax, scenario in zip(axes, scenarios):
        style(ax)
        for y, name in enumerate(order):
            v = np.array([r["attacks_fully_stopped"] for r in robust[scenario][name]]) * 100
            mean, std = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            ax.barh(y, mean, height=0.62, color=COLORS[name], edgecolor=SURFACE, linewidth=2)
            if std > 0:
                ax.errorbar(mean, y, xerr=std, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=0)
            ax.text(min(mean + std, 100) + 2, y, f"{mean:.0f}%", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
        ax.set_title(titles.get(scenario, scenario), loc="left", fontsize=10, color=INK)
    axes[0].set_yticks(range(len(order)), [LABELS[n] for n in order], color=INK, fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Attack sessions fully stopped, by scenario", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out)


def main():
    parser = argparse.ArgumentParser(description="Draw the benchmark charts.")
    parser.add_argument("--seeds", type=Path, default=Path("reports/seed_results.json"))
    parser.add_argument("--robustness", type=Path, default=Path("reports/robustness.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = json.loads(args.seeds.read_text())
    chart_seed_returns(seeds, args.out / "seed_returns.png")
    chart_tradeoff(seeds, args.out / "security_vs_friction.png")
    if args.robustness.exists():
        chart_robustness(json.loads(args.robustness.read_text()), args.out / "robustness.png")


if __name__ == "__main__":
    main()